import torch

def generate(
    model,
    img,
    text,
    args=None
):
    model.eval()

    with torch.no_grad():
        if args.encoder_type in ['ctranspath','swin_tiny']:
            img = model.encoder(img, lora_config=(0.0, args.lora_alpha))
        else:
            img = model.encoder(img)[1]
        img = model.projector(img)                  # bs, project_dim
        if args.decoder_type == 'd_plip':
            img = img.reshape(img.shape[0], -1, 512)    # bs, project_dim//512, 512
        elif args.decoder_type == 'gpt2':
            img = img.reshape(img.shape[0], -1, 768)
        else:
            raise ValueError("Wrong decoder type")
        token = model.tokenizer(text, return_tensors="pt", padding=True)   # bs, seq_len
        if args.decoder_type == 'd_plip':
            input_ids=token['input_ids'][:,:-1].to(args.device)    # skip the eos token
        elif args.decoder_type == 'gpt2':
            input_ids=token['input_ids'].to(args.device) 

        for _ in range(args.generate_length+1):
            if args.decoder_type == 'd_plip':
                attention_mask=torch.where(input_ids<49407,1,0).to(args.device)    # skip the eos token
            elif args.decoder_type == 'gpt2':
                attention_mask=torch.where(input_ids<50257,1,0).to(args.device) 
            output = model.forward_decoder(proj_encoder_feature=img,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask)
            logits = model.decoder_head(output.last_hidden_state[:,-1,:])    # forward the last token embedding though a head, bs x 49408
            
            # Get a token with highest prob, and decode to get a corresponding next word
            next_token = torch.argmax(logits, -1).unsqueeze(1)               # bs x 1
                       # bs x 1

            # Append a next word to current text
            input_ids = torch.cat((input_ids,next_token),dim=1)
    result = model.tokenizer.batch_decode(input_ids)
    for i in range(len(result)):
        if args.dataset == 'luad' and args.type == 'lora':
            predict = result[i].split(' is ')[-1].split(' ')[0]
            result[i] = f"the type of this lung patch is {predict}"
        result[i] = result[i].split('.')[0].replace('<|startoftext|>', '').replace('<|endoftext|>', '') + '.'
        result[i] = result[i].replace(' - ', '-')
    return list(result)

def generate_wsi(
    model,
    img,
    text,
    args=None,
):
    model.eval()

    with torch.no_grad():
        # if args.encoder_type in ['ctranspath','swin_tiny']:
        #     img = model.encoder(img, lora_config=(0.0, args.lora_alpha))
        # else:
        #     img = model.encoder(img)[1]
        img = model.aggregator(img)
        img = model.projector(img)                  # bs, project_dim
        if args.decoder_type == 'd_plip':
            img = img.reshape(img.shape[0], -1, 512)    # bs, project_dim//512, 512
        elif args.decoder_type == 'gpt2':
            img = img.reshape(img.shape[0], -1, 768)
        else:
            raise ValueError("Wrong decoder type")
        token = model.tokenizer(text, return_tensors="pt", padding=True)   # bs, seq_len
        if args.decoder_type == 'd_plip':
            input_ids=token['input_ids'][:,:-1].to(args.device)    # skip the eos token
        elif args.decoder_type == 'gpt2':
            input_ids=token['input_ids'].to(args.device) 

        for _ in range(args.generate_length+1):
            if args.decoder_type == 'd_plip':
                attention_mask=torch.where(input_ids<49407,1,0).to(args.device)    # skip the eos token
            elif args.decoder_type == 'gpt2':
                attention_mask=torch.where(input_ids<50257,1,0).to(args.device) 
            output = model.forward_decoder(proj_encoder_feature=img,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask)
            logits = model.decoder_head(output.last_hidden_state[:,-1,:])    # forward the last token embedding though a head, bs x 49408
            
            # Get a token with highest prob, and decode to get a corresponding next word
            next_token = torch.argmax(logits, -1).unsqueeze(1)               # bs x 1
                       # bs x 1

            # Append a next word to current text
            input_ids = torch.cat((input_ids,next_token),dim=1)
    result = model.tokenizer.batch_decode(input_ids)
    for i in range(len(result)):
        result[i] = result[i].split('.')[0].replace('<|startoftext|>', '').replace('<|endoftext|>', '')
        result[i] = result[i].replace(' - ', '-')
        result[i] = result[i].strip()
        result[i] += '.'
    return list(result)
