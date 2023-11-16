import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8  # VAE_Encoder에서 512 / 8함.
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str, # Negative prompt or empty string
    input_image=None,
    strength=0.8, # add noise proportionally  this paramter value
    do_cfg=True, # Option for Classifier Free Guidence
    cfg_scale=7.5, # it indicates how much we want to pay attention to the coniditioned output with respect to the unconditional output which also means that how much we want the model to pay attention to the condition that we have specified what is the condition The Prompt 
    sampler_name="ddpm", 
    n_inference_steps=50, 
    models={}, 
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    # uncond_prompt는 negative prompt 즉, 정답과 일치하지 않는 prompt라고 함.
    # strength는 입력 이미지와 생성된 이미지 사이에 유사도를 말함. 이게 높으면 입력 이미지와 아예 다른 이미지가 만들어지고, 낮으면 입력 이미지와 유사한 이미지가 생성됨.
    # do_cfg : do classifier free guidance, do_cfg는 모델이 분류자 없이 가이던스를 수행할지 여부를 결정하는 부울 플래그입니다. 이 옵션이 활성화되면, 모델은 분류자(classifier)를 사용하지 않고도 주어진 프롬프트에 대해 이미지를 생성하려고 시도합니다. 이는 모델이 특정 키워드나 개념에 집중하여 더 관련성 높은 이미지를 생성하게 도와줍니다.
    # cfg_scale : How much we want the model to pay attention to our prompt. 분류자 없는 가이던스(classifier-free guidance)의 정도를 결정하는 스케일링 팩터입니다. 이 값이 높을수록 모델은 주어진 텍스트 프롬프트에 더 많이 의존하여 이미지를 생성합니다. 반대로, 낮은 값은 모델이 데이터셋의 분포를 더 자유롭게 따르게 하여, 프롬프트에 덜 구속되는 이미지를 생성하게 합니다.
    # TODO: idle_device가 무슨 뜻인지 이해하기. -> 현재 작업중이지 않은 device(CPU or GPU)
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

    # TODO : torch.Generator가 무엇인지 이해하기 -> torch.Generator(device=device)는 PyTorch 라이브러리에서 사용되는 코드의 일부로, 난수 생성기(Generator) 객체를 생성하는 구문입니다. torch.Generator는 PyTorch에서 난수를 생성할 때 일관된 또는 반복 가능한 결과를 얻기 위해 사용됩니다. 이 난수 생성기는 특정한 시작점(seed)을 설정함으로써 동일한 난수 시퀀스를 재현할 수 있게 해줍니다.
    generator = torch.Generator(device=device)
    if seed is None:
        generate.seed()
    else:
        generator.maual_seed(seed)

    clip = models['clip']
    clip.to(device)
    
    if do_cfg:
        # Convert the prompt into tokens using tokenizer
        cond_tokens = tokenizer.batch_encode_plus((prompt), padding= "max_length", max_length=77).input_ids
        # (Batch_Size, Seq_Len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim(=768))
        # 아래의 코드를 이용해 conditional한 context의 token sequence를 얻음.
        cond_context = clip(cond_tokens)
        
        # if we don't want specify, we will use the empty string
        uncond_tokens = tokenizer.batch_encode_plus((uncond_prompt), padding="max_length", max_length=77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim(=768))
        uncond_context = clip(uncond_tokens)
        
        # (2, Seq_Len, Dim) -> (2, 77, 768)
        context = torch.cat([cond_context, uncond_context])
    else:
        # 만약 CFG를 사용하지 않는 다면 1번의 step through과정만 하면 된다, 즉 context를 combine하는 작업없이 그냥 진행하면 됨.
        # 하지만 model에게 prompt에 얼마나 attention을 해야할 지 결합할 수 없다.
        # TODO : 위 말의 근거를 생각하기.
        tokens = tokenizer.batch_encode_plus((prompt), padding= "max_length", max_length=77).input_ids
        token = torch.Tensor(tokens, dtype=torch.long, device=device)
        # (1, 77, 768)
        context = clip(tokens)
    # since we have finishfed using the clip, we can move ti to the idle device
    # this means move clip model to idle(현재 사용하지 않는) device(gpu -> cpu)
    to_idle(clip)
    
    if sampler_name == 'ddpm':
        sampler = DDPMSampler(generator)
        sampler.set_inference_stpes(n_inference_steps)
    else:
        raise ValueError(f"Unknown sampler {sampler_name}")
    
    latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
    
    # TODO: 학습 과정은 image to image인 것 같다. 정확히 한 번 더 살펴봐야할 듯
    
    if input_image: # Image to Image Pipelline
        encoder = models["encoder"]
        encoder.to(device)
        
        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        input_image_tensor = np.array(input_image_tensor)
        # (Height, Width, Channel)
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
        # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
        input_image_tensor = input_image_tensor.unsqueeze(0)
        # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
        
        # Add Noise
        # TODO: 이후에 적용된 Gaussian Noise를 prediction해서 denoising을 할 예정 -> 이 말이 맞는지 확인 해야함.
        # 위에서 generator에 seed를 적용했는데, 이게 output을 deterministic하게 만듦.
        encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
        # run the image thorugh of the VAE
        # TODO: 지금 여기서 noise한 image를 만들기 위해 noise를 더했는데, 왜 아래에서 sampler에 strength를 더하면서 add_noise를 하고 있는 것인가...?
        latents = encoder(input_image_tensor, encoder_noise)
        
        # strength paramter는 input image에 얼마나 attention할 것인지를 정하는 hypter parameter입니다. 
        # 직관적으로는 더해질 noise의 strength.
        # strength가 높으면 noise가 더 많이 더해질 것이고, 이는 더 creative한 image가 생성될 것입니다. 
        # TODO: 그 이유는 더 많은 정보가 소실될 것이고, 이는 더 많은 noise를 지우게 될 것이므로, 분포가 더 variety해 질 것으로 예상됩니다.
        # TODO: 반대로 생각하면 noise가 적으면 원본 이미지가 그대로 복원될 것이라고 직관적으로 생각되기 때문.
        # TODO: 지금 여기서 noise한 image를 만들기 위해 noise를 더했는데, 왜 아래에서 sampler에 strength를 더하면서 add_noise를 하고 있는 것인가...?
        # strength는 initial noise level을 정한다...? 그래서 1로 설정하면 maximum한 noise level에서 시작한다.
        # TODO: 그럼 0이면 아예 없는건가?
        # strength level에 의해서 encoded image에 noise가 더해진다.
        sampler.set_strength(strength=strength)
        latent = sampler.add_noise(latents, sampler.timesteps[0])
        
        to_idle(encoder)
    else: # Text to Image Pipeline
        # If we are doing text-to-text, start with random noise from N(0, 1)
        latents = torch.randn(latent_shape, generator=generator, device=device)
        
    diffusion = models["diffusion"]
    diffusion.to(device)
    
    # 현재 50번의 n_inference_steps을 가지고 있으므로 이를 1000번에 적용해서 각각의 step을 뽑아내면 
    # 1000, 980, 960, ... 20, 0이 될 것이다.
    # 각각의 step은 noise level을 의미하는 것이고, 
    # 우리는 scheduler에게 위의 time step에 해당하는 denoising inference step을 지정해 줄 수 있다.
    
    timesteps = tqdm(sampler.timsteps)
    for i, timestep in enumerate(timesteps):
        # timestep(1) -> time_embedding(1, 320)
        time_embedding = get_time_embedding(timestep).to(device)
        
        # (Batch_Size, 4, Latents_Height, Latents_Width)
        model_input = latents
        
        # 3:39.30
        if do_cfg:
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = model_input.repeat(2, 1, 1, 1)
        
        # model_output is the predicted noise by the UNET
        model_output = diffusion(model_input, context, time_embedding)
        
        if do_cfg:
            output_cond, output_uncond = model_output.chunk(2)
            # CFG의 formular에 의해 다음과 같은 코드가 필요하게 된다.
            # output = w * (output_cond - output_uncond) + output_uncond
            model_output = cfg_scale * (output_cond-output_uncond) + output_uncond
        
        # Text to image
        # UNET will predict the noise in the latents
        # remove noise from the image to a less noisy image. This is done by the scheduler
        # so at each step we ask the UNET, how much noise is in the image. We renove it and then 
        # we give it again to the UNET and ask how much noise is there and we remove it.  -> 위 과정을 반복
        # 이후 얻은 latent를 decoder로 보내서 image를 생성함. 
        
        # Remove noise predicted by the UNET
        latents = sampler.step(timestep, latents, model_output)
    
    to_idle(diffusion)
    
    decoder = models["decoder"]
    decoder.to(device)
    
    images = decoder(latents)
    to_idle(decoder)
    
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
    images = images.permute(0, 2, 3, 1)
    images = images.to('cpu', torch.uint8)
    
    return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
    return x    

def get_time_embedding(timestep):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    # TODO: 아래의 문법 이해하기. 왜 None이 들어가노;
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)