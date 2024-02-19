import pdb
import transformers
import torch
import torch.nn.functional as F
transformers.logging.set_verbosity_error()

from pdb import Pdb
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from torch import kl_div, nn

from models.blip import create_vit, init_tokenizer

class DLIPPretrainSigmoid(nn.Module):
    def __init__(self,
                 med_config='configs/bert-small_config.json',
                 image_size=224,
                 vit='small',
                 bert='small',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                 ):

        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        print(f'vit arch: {vit}')
        if vit == 'base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        elif vit == 'small':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        elif vit == 'tiny':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        elif vit == 'large':
            print(vit)
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder, default_cfgs['vit_large_patch16_224_in21k'])
        elif vit == 'mid':
            checkpoint = torch.load('/home/tiger/cache/BLIP/vit_mid.pth')
            state_dict = checkpoint["state_dict"]
            pt_state_dict = {}
            for k1, k2 in zip(self.visual_encoder.state_dict().keys(), state_dict.keys()):
                print(k1, k2)
                pt_state_dict[k1] = state_dict[k2]
            msg = self.visual_encoder.load_state_dict(pt_state_dict, strict=False)
            print(msg)
        else:
            pass
        total = sum([param.nelement() for param in self.visual_encoder.parameters()])
        print('visial student  Number of params: %.2fM' % (total / 1e6))

        print(f'bert arch: {bert}')
        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.config = encoder_config
        if bert == 'base':
            self.text_encoder = BertModel.from_pretrained('bert-base-cased', config=encoder_config,
                                                          ignore_mismatched_sizes=True, add_pooling_layer=False)
        elif bert == 'small':
            self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
            # self.text_encoder = BertModel.from_pretrained('prajjwal1/bert-medium',config=encoder_config, ignore_mismatched_sizes=True,add_pooling_layer=False)
        elif bert == 'mid':
            self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        elif bert == 'tiny':
            self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
            # self.text_encoder = BertModel.from_pretrained('prajjwal1/bert-mini',config=encoder_config, ignore_mismatched_sizes=True,add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        text_width = self.text_encoder.config.hidden_size

        total = sum([param.nelement() for param in self.text_encoder.parameters()])
        print('text student Number of params: %.2fM' % (total / 1e6))

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2)
        self.vl_proj = nn.Linear(text_width, 768)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.criterion_kl = nn.KLDivLoss(reduction='sum')

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        if bert == 'base':
            self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)
        elif bert == 'small':
            self.text_decoder = BertLMHeadModel(config=decoder_config)
        elif bert == 'mid':
            self.text_decoder = BertLMHeadModel(config=decoder_config)
            # self.text_decoder = BertLMHeadModel.from_pretrained('prajjwal1/bert-medium',config=decoder_config)
        elif bert == 'tiny':
            self.text_decoder = BertLMHeadModel(config=decoder_config)
            # self.text_decoder = BertLMHeadModel.from_pretrained('prajjwal1/bert-mini',config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')

        # Sigmoid Loss
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

    def forward(self, image, caption, teacher, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        bs = image.size(0)
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30,
                              return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,
                                       output_attentions = True,
                                       return_dict = True,
                                      )            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        result = text_feat @ image_feat.T * self.logit_scale.exp() + self.logit_bias

        # vl_output = self.itm_head(vl_embeddings)

        labels = 2 * torch.eye(bs).to(image.device) - torch.ones(bs).to(image.device)
        loss_itm = -torch.sum(F.logsigmoid(labels * result)) / bs

        ##================= LM ========================##
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        decoder_output = self.text_decoder(decoder_input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           output_attentions=True,
                                           output_hidden_states=True,
                                           )

        loss_lm = decoder_output.loss
        prediction_scores = decoder_output.logits
        last_states_decoder = decoder_output.hidden_states[-1][:, 0, :]

        ###============== Dist Loss  ===================###
        # get teacher features
        with torch.no_grad():
            image_embeds_tea = teacher.visual_encoder(image)
            image_atts_tea = torch.ones(image_embeds_tea.size()[:-1], dtype=torch.long).to(image.device)

            image_feat_tea = F.normalize(teacher.vision_proj(image_embeds_tea[:, 0, :]), dim=-1)

            image_feat_all = torch.cat([image_feat_tea.t(), self.image_queue.clone().detach()], dim=1)

            text_output_tea = teacher.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                                   return_dict=True, mode='text')
            text_feat_tea = F.normalize(teacher.text_proj(text_output_tea.last_hidden_state[:, 0, :]), dim=-1)




            encoder_output_tea = teacher.text_encoder(encoder_input_ids,
                                                      attention_mask=text.attention_mask,
                                                      encoder_hidden_states=image_embeds_tea,
                                                      encoder_attention_mask=image_atts_tea,
                                                      return_dict=True,
                                                      output_attentions=True,
                                                      output_hidden_states=True,
                                                      )

            decoder_output_tea = teacher.text_decoder(decoder_input_ids,
                                                      attention_mask=text.attention_mask,
                                                      encoder_hidden_states=image_embeds_tea,
                                                      encoder_attention_mask=image_atts_tea,
                                                      labels=decoder_targets,
                                                      return_dict=True,
                                                      output_attentions=True,
                                                      output_hidden_states=True,
                                                      )
            prediction_scores_tea = decoder_output_tea.logits
            last_states_decoder_tea = decoder_output_tea.hidden_states[-1][:, 0, :]

            ###============== Dist CLIP  ===================###
            ##=======  cosine similar loss  =======##
            result_teacher = text_feat_tea @ image_feat_tea.T * self.logit_scale.exp() + self.logit_bias
            loss_ita_dist = self.criterion_kl(F.logsigmoid(labels * result_teacher), F.sigmoid(labels * result))

        criterion_cos = nn.CosineEmbeddingLoss(0.25)
        bs = image.size(0)
        t = torch.ones(bs).to(image.device)
        loss_img_dist = criterion_cos(image_feat, image_feat_tea, t)
        loss_text_dist = criterion_cos(text_feat, text_feat_tea, t)

        vl_feat_s = self.text_proj(output_pos.last_hidden_state[:, 0, :])
        vl_feat_t = teacher.text_proj(encoder_output_tea.last_hidden_state[:, 0, :])
        loss_vl_dist = criterion_cos(vl_feat_s, vl_feat_t, t)

        vl_decoder_s = self.text_proj(last_states_decoder)
        vl_decoder_t = teacher.text_proj(last_states_decoder_tea)
        loss_vl_dec_dist = criterion_cos(vl_decoder_s, vl_decoder_t, t)

        ##======= logits KL loss  =======##a
        temp = 1.0
        vocab_size = int(self.config.vocab_size)
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        logits = shifted_prediction_scores.view(-1, vocab_size)

        shifted_prediction_scores_tea = prediction_scores_tea[:, :-1, :].contiguous()
        logits_tea = shifted_prediction_scores_tea.view(-1, vocab_size)
        bs_logits = logits.size(0)
        loss_kl = (1.0 / bs_logits) * self.criterion_kl(F.log_softmax(logits / temp, dim=1),
                                                        F.softmax(logits_tea / temp, dim=1))

        ##======= Self Attentions loss  =======##
        temp_score = 0.05
        attention = output_pos.attentions[-1]
        attention_tea = encoder_output_tea.attentions[-1]
        attention_score = attention.view(-1, 30)
        attention_score_tea = attention_tea.view(-1, 30)
        bs_atten = attention_score.size(0)
        loss_Sattn = (1.0 / bs_atten) * self.criterion_kl(F.log_softmax(attention_score / temp_score, dim=1),
                                                          F.softmax(attention_score_tea / temp_score, dim=1))

        ##======= Cross Attentions loss  =======##
        temp_score = 0.05
        x_attention = output_pos.cross_attentions[-1]
        x_attention_tea = encoder_output_tea.cross_attentions[-1]
        x_attention_score = x_attention.view(-1, 197)
        x_attention_score_tea = x_attention_tea.view(-1, 197)
        bs_atten = x_attention_score.size(0)
        loss_Xattn = (1.0 / bs_atten) * self.criterion_kl(F.log_softmax(x_attention_score / temp_score, dim=1),
                                                          F.softmax(x_attention_score_tea / temp_score, dim=1))

        ##======= Decoder Self Attentions loss  =======##
        temp_score = 0.05
        dx_attention = decoder_output.attentions[-1]
        dx_attention_tea = decoder_output_tea.attentions[-1]
        dx_attention_score = dx_attention.view(-1, 30)
        dx_attention_score_tea = dx_attention_tea.view(-1, 30)
        bs_atten = dx_attention_score.size(0)
        loss_DXattn = (1.0 / bs_atten) * self.criterion_kl(F.log_softmax(dx_attention_score / temp_score, dim=1),
                                                           F.softmax(dx_attention_score_tea / temp_score, dim=1))

        loss_dist = 0
        loss_dist += loss_img_dist
        loss_dist += loss_text_dist
        loss_dist += loss_vl_dist
        loss_dist += loss_vl_dec_dist
        loss_dist += loss_ita_dist
        loss_dist += loss_kl
        loss_dist += loss_Sattn
        loss_dist += loss_Xattn
        loss_dist += loss_DXattn

        return torch.tensor(0), loss_itm, loss_lm, loss_dist

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # # gather keys before updating queue
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)
        image_feats = image_feat
        text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


def dlip_pretrain_sigmoid(**kwargs):
    model = DLIPPretrainSigmoid(**kwargs)
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
    # return tensor


from typing import List


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            skip_key: str,
            depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + ' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                    len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)
