import torch
import torch.nn as nn
from transformers import AutoProcessor
from .swin_transformer import ctranspath
from .clip import CLIPModel
from .prompt import EncoderPrompt, DecoderPrompt
# from model.projector import MLP

class PromptModel(nn.Module):
    def __init__(self, 
                 encoder_type='ctranspath',
                 enconder_prompt_len=1,
                 enconder_ckpt_path='/home/compu/anhnguyen/TransPath/ctranspath.pth',
                 enconder_skip_layers=[],
                 decoder_type='plip',
                 decoder_prompt_len=1,
                 decoder_ckpt_path='/home/compu/anhnguyen/prompt_works/plip_ckpt/',
                 decoder_skip_layers=[],
                 tokenizer_type="vinid/plip",
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Init visual encoder
        if encoder_type == 'ctranspath':
            self.encoder = ctranspath()
            self.encoder.head = nn.Identity()
            td = torch.load(enconder_ckpt_path)
            self.encoder.load_state_dict(td['model'], strict=True)
        else:
            raise ValueError(f'Encoder {encoder_type} is not supported')
        
        # TODO: Finalize Init projector
        layers = []
        sizes = [768, 1024, 512]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
            if i < len(sizes) - 2:
                layers.append(nn.GELU())
        self.projector = nn.Sequential(*layers)

        # Init text decoder
        if decoder_type == 'plip':
            self.decoder = CLIPModel.from_pretrained(decoder_ckpt_path).text_model
            self.decoder_head = nn.Linear(512, 49408)
        else:
            raise ValueError(f'Decoder {decoder_type} is not supported')

        self.freeze_encoder_and_decoder()

        # Init prompt for encoder
        self.encoder_prompt = EncoderPrompt(encoder_type, enconder_prompt_len, enconder_skip_layers)
        self.key, self.encoder_prompt_dict = self.encoder_prompt.prompt_combination
        for layer_id in self.encoder_prompt_dict:
            setattr(self.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
        
        # Init prompt for decoder
        self.decoder_prompt = DecoderPrompt(decoder_type, decoder_prompt_len, decoder_skip_layers)
        self.decoder_prompt_dict = self.decoder_prompt.prompt_combination
        for layer_id in self.decoder_prompt_dict:
            setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])

        # Init tokenizer
        self.tokenizer = AutoProcessor.from_pretrained(tokenizer_type)

    def freeze_encoder_and_decoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_query(self, x):
        with torch.no_grad():
            query = self.encoder(x, use_prompt=False)
        return query

    def forward(self, img, text):
        assert len(img.shape)==4 and img.shape[1:] == (3,224,224), \
                    f'Expect img input of shape (bs,3,224,3,224) but got {img.shape}'
        img = self.encoder(img, use_prompt=True)
        img = self.projector(img)
        img = img.reshape(img.shape[0], -1, 512)

        assert isinstance(text,list), f'Expect text input is a list, but got {type(text)}'
        assert len(text) == img.shape[0], \
                    f'Expect two inputs have same length, but got img of {img.shape[0]} and text of {len(text)}'
        text = self.tokenizer(text, return_tensors="pt", padding=True)
        output = self.decoder(proj_encoder_feature=img,
                              input_ids=text['input_ids'], 
                              attention_mask=text['attention_mask'])
        output = self.decoder_head(output.pooler_output)

        return output
    
    def forward_decoder(self, inputs, attention_masks):
        x = self.decoder(input_ids=inputs, attention_mask=attention_masks) # TODO: when implemeted with input_embed, change this to input_embed
     
        return x
