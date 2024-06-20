import torch
import torch.nn as nn
from transformers import AutoProcessor, GPT2Tokenizer

from .clip import CLIPModel
from .gpt2 import GPT2Model
from .prompt import EncoderPrompt, DecoderPrompt, Lora
from .projector import MLP, MLP_for_prompt
from .r2t_rrt import RRTEncoder
from .r2t_transmil import TransMIL
from .rrt_ibmil import Dattention_ori
from .r2t_clam import CLAM_MB, CLAM_SB

class PromptModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        # Init model components
        self._init_encoder()
        self._init_projector()
        self._init_decoder()
        if args.type not in ['full_ft','single_encoder']:
            self._freeze_encoder_and_decoder()
            self._init_prompt()
        self._init_tokenizer()

    def _init_encoder(self):
        if self.args.encoder_type == 'ctranspath':
            from .swin_transformer import ctranspath, swinv1
            self.encoder = ctranspath()
            self.encoder.head = nn.Identity()
            td = torch.load(self.args.encoder_ckpt_path)
            self.encoder.load_state_dict(td['model'], strict=True)
        elif self.args.encoder_type == 'swin_tiny':
            from .swin_transformer import swinv1
            self.encoder = swinv1()
            td = torch.load(self.args.encoder_ckpt_path)
            self.encoder.load_state_dict(td['model'], strict=True)
            self.encoder.head = nn.Identity()
        elif self.args.encoder_type == 'e_plip':
            self.encoder = CLIPModel.from_pretrained(self.args.decoder_ckpt_path).vision_model
            self.encoder.encoder.decoder_skip_layers_for_visual = [i for i in range(12)]
        
        else:
            raise ValueError(f'Encoder {self.args.encoder_type} is not supported')

    def _init_decoder(self):
        if self.args.decoder_type == 'd_plip':
            self.decoder = CLIPModel.from_pretrained(self.args.decoder_ckpt_path).text_model
            self.decoder_head = nn.Linear(512, 49408)
            self.decoder.encoder.decoder_skip_layers_for_visual = self.args.decoder_skip_layers_for_visual
        elif self.args.decoder_type == 'gpt2':
            self.decoder = GPT2Model.from_pretrained(self.args.decoder_ckpt_path)
            self.decoder_head = nn.Linear(768, 50258) 
            self.decoder.decoder_skip_layers_for_visual = self.args.decoder_skip_layers_for_visual
        else:
            raise ValueError(f'Decoder {self.args.decoder_type} is not supported')

    def _init_prompt(self):
        if self.args.type != 'lora':
            # Init prompt for encoder
            self.encoder_prompt = EncoderPrompt(self.args.encoder_type, 
                                                self.args.encoder_prompt_len, 
                                                self.args.encoder_skip_layers,
                                                True if self.args.type == 'distinct' else False)
            self.key, self.encoder_prompt_dict = self.encoder_prompt.prompt_combination
            if self.args.encoder_type == 'ctranspath':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id]) 
            elif self.args.encoder_type == 'e_plip':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
            # Init prompt for decoder
            self.decoder_prompt = DecoderPrompt(self.args.decoder_type, self.args.decoder_prompt_len, self.args.decoder_skip_layers)
            self.decoder_prompt_dict = self.decoder_prompt.prompt_combination[1]
            for layer_id in self.decoder_prompt_dict:
                if self.args.decoder_type == 'd_plip':
                    setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])
                elif self.args.decoder_type == 'gpt2':
                    setattr(self.decoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])
        elif self.args.type == 'lora':
            self.encoder_lora = Lora(self.args, module='encoder')
            self.key, self.encoder_lora_dict = self.encoder_lora.lora_combination
            self.decoder_lora = Lora(self.args, module='decoder')
            self.decoder_lora_dict = self.decoder_lora.lora_combination[1]

            for layer_id in self.encoder_lora_dict:
                if self.args.encoder_type in ['ctranspath','swin_tiny']:
                    setattr(self.encoder, f'lora_layer_{layer_id}', self.encoder_lora_dict[layer_id])
                elif self.args.encoder_type == 'e_plip':
                    setattr(self.encoder.encoder, f'lora_layer_{layer_id}', self.encoder_lora_dict[layer_id])

            for layer_id in self.decoder_lora_dict:
                if self.args.decoder_type == 'd_plip':
                    setattr(self.decoder.encoder, f'lora_layer_{layer_id}', self.decoder_lora_dict[layer_id])
                elif self.args.decoder_type == 'gpt2':
                    setattr(self.decoder, f'lora_layer_{layer_id}', self.decoder_lora_dict[layer_id])

    def _init_tokenizer(self):
        if self.args.tokenizer_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.tokenizer_type)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.decoder.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoProcessor.from_pretrained(self.args.tokenizer_type)

    def _init_projector(self):
        self.projector = MLP(self.args)

    def _freeze_encoder_and_decoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_query(self, img, text):
        with torch.no_grad():
            if self.args.encoder_type in ['ctranspath','swin_tiny']:
                visual_query = self.encoder(img, use_prompt=False, use_lora=False, lora_config=None)
                token = self.tokenizer(text, return_tensors="pt", padding=True)
                input_ids=token['input_ids'][:,:-1].to(self.args.device)
                attention_mask=torch.where(input_ids<50257,1,0).to(self.args.device)
                if self.args.decoder_type == 'd_plip':
                    text_query = self.decoder(input_ids=input_ids, 
                                          proj_encoder_feature=None, 
                                          attention_mask=attention_mask,
                                          use_prompt=False,
                                          use_lora=False, 
                                          lora_config=None).pooler_output
                elif self.args.decoder_type == 'gpt2':
                    text_query = self.decoder(input_ids=input_ids, 
                                          proj_encoder_feature=None, 
                                          attention_mask=attention_mask,
                                          use_prompt=False,
                                          use_lora=False, 
                                          lora_config=None).last_hidden_state[:,-1,:]
                query = torch.cat((visual_query, text_query),dim=1)
            elif self.args.encoder_type == 'e_plip':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', None)
                query = self.encoder(img)[1]
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
        return query

    def forward(self, img, text='The cancer grade of this colon patch is'):
        assert len(img.shape)==4 and img.shape[1:] == (3,224,224), \
                    f'Expect img input of shape (bs,3,224,3,224) but got {img.shape}'
        # Forward through an encoder
        if self.args.encoder_type in ['ctranspath','swin_tiny']:
            img = self.encoder(img, lora_config=(self.args.lora_drop_out, self.args.lora_alpha))
        elif self.args.encoder_type == 'e_plip':
            img = self.encoder(img)[1]

        # Forward through a projector
        img = self.projector(img)
        if self.args.decoder_type == 'd_plip':
            img = img.reshape(img.shape[0], -1, 512)
        elif self.args.decoder_type == 'gpt2':
            img = img.reshape(img.shape[0], -1, 768)

        # Forward though a decoder
        text = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = text['input_ids'].to(self.device)
        attention_mask = text['attention_mask'].to(self.device)
        # attention_mask = torch.tensor([[1]*input_ids.shape[1]]).to(self.device)

        output = self.decoder(proj_encoder_feature=img, 
                              input_ids=input_ids, 
                              attention_mask=attention_mask,
                              lora_config=(self.args.lora_drop_out, self.args.lora_alpha)
                              )
        logits = self.decoder_head(output.last_hidden_state)

        return {
            'input_ids': input_ids,
            'last_layer_logits': logits
        }
    
    def forward_decoder(self, proj_encoder_feature, input_ids, attention_mask):
        output = self.decoder(proj_encoder_feature=proj_encoder_feature,
                              input_ids=input_ids, 
                              attention_mask=attention_mask,
                              lora_config=(0.0, self.args.lora_alpha),
                              output_attentions=True)
        return output

class AttentionGated(nn.Module):
    def __init__(self,input_dim,act='relu',num_class=None,bias=False,dropout=False,rrt=None,head=None):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(input_dim, 768)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        if rrt is not None:
            self.feature += [rrt] 
        self.feature = nn.Sequential(*self.feature)

        if head is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.L*self.K, num_class),
            )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.head = head


    def forward(self, x):
        x = self.feature(x.squeeze(0))
        # x = self.bnorm(x)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = nn.functional.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)  # 1, dim

        if self.head is not None:
            x = self.classifier(x)

        return x

class MIL_caption(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        # Init model components
        self._init_projector()
        self._init_decoder()
        self._freeze_decoder()
        self._init_prompt()
        self._init_tokenizer()
        self.aggregator = AttentionGated(768)
        if args.module == 'rrt':
            self.module = RRTEncoder(mlp_dim=768, epeg_k=15,crmsa_k=3)

    def _init_decoder(self):
        if self.args.decoder_type == 'd_plip':
            self.decoder = CLIPModel.from_pretrained(self.args.decoder_ckpt_path).text_model
            self.decoder_head = nn.Linear(512, 49408)
            self.decoder.encoder.decoder_skip_layers_for_visual = self.args.decoder_skip_layers_for_visual
        elif self.args.decoder_type == 'gpt2':
            self.decoder = GPT2Model.from_pretrained(self.args.decoder_ckpt_path)
            self.decoder_head = nn.Linear(768, 50258) 
            self.decoder.decoder_skip_layers_for_visual = self.args.decoder_skip_layers_for_visual
        else:
            raise ValueError(f'Decoder {self.args.decoder_type} is not supported')

    def _init_prompt(self):
        if self.args.type != 'lora':
            # Init prompt for encoder
            self.encoder_prompt = EncoderPrompt(self.args.encoder_type, 
                                                self.args.encoder_prompt_len, 
                                                self.args.encoder_skip_layers,
                                                True if self.args.type == 'distinct' else False)
            self.key, self.encoder_prompt_dict = self.encoder_prompt.prompt_combination
            if self.args.encoder_type == 'ctranspath':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id]) 
            elif self.args.encoder_type == 'e_plip':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
            # Init prompt for decoder
            self.decoder_prompt = DecoderPrompt(self.args.decoder_type, self.args.decoder_prompt_len, self.args.decoder_skip_layers)
            self.decoder_prompt_dict = self.decoder_prompt.prompt_combination[1]
            for layer_id in self.decoder_prompt_dict:
                if self.args.decoder_type == 'd_plip':
                    setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])
                elif self.args.decoder_type == 'gpt2':
                    setattr(self.decoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])
        elif self.args.type == 'lora':
            # self.key, self.encoder_lora_dict = self.encoder_lora.lora_combination
            self.decoder_lora = Lora(self.args, module='decoder')
            self.decoder_lora_dict = self.decoder_lora.lora_combination[1]

            for layer_id in self.decoder_lora_dict:
                if self.args.decoder_type == 'd_plip':
                    setattr(self.decoder.encoder, f'lora_layer_{layer_id}', self.decoder_lora_dict[layer_id])
                elif self.args.decoder_type == 'gpt2':
                    setattr(self.decoder, f'lora_layer_{layer_id}', self.decoder_lora_dict[layer_id])

    def _init_tokenizer(self):
        if self.args.tokenizer_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.tokenizer_type)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.decoder.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoProcessor.from_pretrained(self.args.tokenizer_type)

    def _init_projector(self):
        self.projector = MLP(self.args)

    def _freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_query(self, img, text):
        with torch.no_grad():
            if self.args.encoder_type in ['ctranspath','swin_tiny']:
                visual_query = torch.mean(img, dim=0).unsqueeze(0)
                token = self.tokenizer(text, return_tensors="pt", padding=True)
                input_ids=token['input_ids'][:,:-1].to(self.args.device)
                attention_mask=torch.where(input_ids<50257,1,0).to(self.args.device)
                if self.args.decoder_type == 'd_plip':
                    text_query = self.decoder(input_ids=input_ids, 
                                          proj_encoder_feature=None, 
                                          attention_mask=attention_mask,
                                          use_prompt=False,
                                          use_lora=False, 
                                          lora_config=None).pooler_output
                elif self.args.decoder_type == 'gpt2':
                    text_query = self.decoder(input_ids=input_ids, 
                                          proj_encoder_feature=None, 
                                          attention_mask=attention_mask,
                                          use_prompt=False,
                                          use_lora=False, 
                                          lora_config=None).last_hidden_state[:,-1,:]
                query = torch.cat((visual_query, text_query),dim=1)
            elif self.args.encoder_type == 'e_plip':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', None)
                query = self.encoder(img)[1]
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
        return query

    def forward(self, img, text):
        # Forward through an encoder
        # if self.args.encoder_type in ['ctranspath','swin_tiny']:
        #     img = self.encoder(img, lora_config=(0.1, 12))
        # elif self.args.encoder_type == 'e_plip':
        #     img = self.encoder(img)[1]
        if self.args.module == 'rrt':
            img = self.module(img)
        img = self.aggregator(img)  # num_patch x dim -> 1 x dim

        # Forward through a projector
        img = self.projector(img)
        if self.args.decoder_type == 'd_plip':
            img = img.reshape(img.shape[0], -1, 512) # 1, 1, 512
        elif self.args.decoder_type == 'gpt2':
            img = img.reshape(img.shape[0], -1, 768)

        # Forward though a decoder
        text = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = text['input_ids'].to(self.device)
        attention_mask = torch.tensor([[1]*input_ids.shape[1]]).to(self.device)
        output = self.decoder(proj_encoder_feature=img, 
                              input_ids=input_ids, 
                              attention_mask=attention_mask,
                              lora_config=(0.1, 12)
                              )
        logits = self.decoder_head(output.last_hidden_state)

        return {
            'input_ids': input_ids,
            'last_layer_logits': logits
        }
    
    def forward_decoder(self, proj_encoder_feature, input_ids, attention_mask):
        output = self.decoder(proj_encoder_feature=proj_encoder_feature,
                              input_ids=input_ids, 
                              attention_mask=attention_mask,
                              lora_config=(0.0, self.args.lora_alpha),
                              output_attentions=True)
        return output

class MIL_index(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        if args.mil_type == 'ab-mil':
            self.aggregator = AttentionGated(768, num_class=args.num_class, head=True)
            if args.module == 'rrt':
                self.module = RRTEncoder(mlp_dim=768, epeg_k=15,crmsa_k=3)
        elif args.mil_type == 'transmil':
            self.module = TransMIL(input_dim=768, n_classes=args.num_class, dropout=False, act='relu')
        elif args.mil_type == 'ibmil':
            self.module = Dattention_ori(out_dim=args.num_class, in_size=768)
        elif args.mil_type == 'ibmil':
            self.module = Dattention_ori(out_dim=args.num_class, in_size=768)
        elif args.mil_type == 'clam_sb':
            self.module = CLAM_SB(input_dim=768, n_classes=args.num_class)
        elif args.mil_type == 'clam_mb':
            self.module = CLAM_MB(input_dim=768, n_classes=args.num_class)

    def get_query(self, img, text):
        with torch.no_grad():
            if self.args.encoder_type in ['ctranspath','swin_tiny']:
                visual_query = torch.mean(img, dim=0).unsqueeze(0)
                token = self.tokenizer(text, return_tensors="pt", padding=True)
                input_ids=token['input_ids'][:,:-1].to(self.args.device)
                attention_mask=torch.where(input_ids<50257,1,0).to(self.args.device)
                if self.args.decoder_type == 'd_plip':
                    text_query = self.decoder(input_ids=input_ids, 
                                          proj_encoder_feature=None, 
                                          attention_mask=attention_mask,
                                          use_prompt=False,
                                          use_lora=False, 
                                          lora_config=None).pooler_output
                elif self.args.decoder_type == 'gpt2':
                    text_query = self.decoder(input_ids=input_ids, 
                                          proj_encoder_feature=None, 
                                          attention_mask=attention_mask,
                                          use_prompt=False,
                                          use_lora=False, 
                                          lora_config=None).last_hidden_state[:,-1,:]
                query = torch.cat((visual_query, text_query),dim=1)
            elif self.args.encoder_type == 'e_plip':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', None)
                query = self.encoder(img)[1]
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
        return query

    def forward(self, img):
        if self.args.mil_type == 'ab-mil':
            if self.args.module == 'rrt':
                img = self.module(img)
            img = self.aggregator(img)  # num_patch x dim -> 1 x dim
        elif self.args.mil_type in ['transmil','ibmil','clam_sb','clam_mb']:
            img = self.module(img)

        return img
