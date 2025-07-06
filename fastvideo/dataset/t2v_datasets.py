import json
import math
import os
import random
from collections import Counter
from os.path import join as opj

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from fastvideo.utils.dataset_utils import DecordInit
from fastvideo.utils.logging_ import main_print

import io
import decord
from torch.utils.data import DataLoader, Dataset
import torch
from diffusers import AutoencoderKLWan
from typing import List, Optional, Tuple, Union
from petrel_client.client import Client
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
from transformers import AutoTokenizer, UMT5EncoderModel
import ftfy
import regex as re
import html


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DataSetProg(metaclass=SingletonMeta):

    def __init__(self):
        self.cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()

    def set_cap_list(self, num_workers, cap_list, n_elements):
        self.num_workers = num_workers
        self.cap_list = cap_list
        self.n_elements = n_elements
        self.elements = list(range(n_elements))
        random.shuffle(self.elements)
        print(f"n_elements: {len(self.elements)}", flush=True)

        for i in range(self.num_workers):
            self.n_used_elements[i] = 0
            per_worker = int(
                math.ceil(len(self.elements) / float(self.num_workers)))
            start = i * per_worker
            end = min(start + per_worker, len(self.elements))
            self.worker_elements[i] = self.elements[start:end]

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][
            self.n_used_elements[worker_id] %
            len(self.worker_elements[worker_id])]
        self.n_used_elements[worker_id] += 1
        return idx


dataset_prog = DataSetProg()


def filter_resolution(h,
                      w,
                      max_h_div_w_ratio=17 / 16,
                      min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False


class T2V_dataset(Dataset):

    def __init__(self, args, transform, temporal_sample, tokenizer,
                 transform_topcrop):
        self.data = args.data_merge_path
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num
        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.text_max_length = args.text_max_length
        self.cfg = args.cfg
        self.speed_factor = args.speed_factor
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.drop_short_ratio = args.drop_short_ratio
        assert self.speed_factor >= 1
        self.v_decoder = DecordInit()
        self.video_length_tolerance_range = args.video_length_tolerance_range
        self.support_Chinese = True
        if "mt5" not in args.text_encoder_name:
            self.support_Chinese = False

        cap_list = self.get_cap_list()

        assert len(cap_list) > 0
        cap_list, self.sample_num_frames = self.define_frame_index(cap_list)
        self.lengths = self.sample_num_frames

        n_elements = len(cap_list)
        dataset_prog.set_cap_list(args.dataloader_num_workers, cap_list,
                                  n_elements)

        print(f"video length: {len(dataset_prog.cap_list)}", flush=True)

    def set_checkpoint(self, n_used_elements):
        for i in range(len(dataset_prog.n_used_elements)):
            dataset_prog.n_used_elements[i] = n_used_elements

    def __len__(self):
        return dataset_prog.n_elements

    def __getitem__(self, idx):
        while True:
            try:
                data = self.get_data(idx)
                return data
            except Exception as e:
                idx += 1
                print(f"Error fetching data at idx={idx-1}, trying idx={idx}: {e}")
        

    def get_data(self, idx):
        path = dataset_prog.cap_list[idx]["path"]
        if path.endswith(".mp4"):
            return self.get_video(idx)
        else:
            return self.get_image(idx)

    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]["path"]
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indices = dataset_prog.cap_list[idx]["sample_frame_index"]
        torchvision_video, _, metadata = torchvision.io.read_video(
            video_path, output_format="TCHW")
        video = torchvision_video[frame_indices]
        video = self.transform(video)
        video = rearrange(video, "t c h w -> c t h w")
        video = video.to(torch.uint8)
        assert video.dtype == torch.uint8

        h, w = video.shape[-2:]
        assert (
            h / w <= 17 / 16 and h / w >= 8 / 16
        ), f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}"

        video = video.float() / 127.5 - 1.0

        text = dataset_prog.cap_list[idx]["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text[0] if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"]
        cond_mask = text_tokens_and_mask["attention_mask"]
        return dict(
            pixel_values=video,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=video_path,
        )

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[
            idx]  # [{'path': path, 'cap': cap}, ...]

        image = Image.open(image_data["path"]).convert("RGB")  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  #  [1 c h w]

        image = (self.transform_topcrop(image) if "human_images"
                 in image_data["path"] else self.transform(image)
                 )  #  [1 C H W] -> num_img [1 C H W]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        image = image.float() / 127.5 - 1.0

        caps = (image_data["cap"] if isinstance(image_data["cap"], list) else
                [image_data["cap"]])
        caps = [random.choice(caps)]
        text = caps
        input_ids, cond_mask = [], []
        text = text[0] if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"]  # 1, l
        cond_mask = text_tokens_and_mask["attention_mask"]  # 1, l
        return dict(
            pixel_values=image,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=image_data["path"],
        )

    def define_frame_index(self, cap_list):
        new_cap_list = []
        sample_num_frames = []
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_resolution_mismatch = 0
        cnt_movie = 0
        cnt_img = 0
        for i in cap_list:
            path = i["path"]
            cap = i.get("cap", None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue
            if path.endswith(".mp4"):
                # ======no fps and duration=====
                duration = i.get("duration", None)
                fps = i.get("fps", None)
                if fps is None or duration is None:
                    continue

                # ======resolution mismatch=====
                resolution = i.get("resolution", None)
                if resolution is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    if (resolution.get("height", None) is None
                            or resolution.get("width", None) is None):
                        cnt_no_resolution += 1
                        continue
                    height, width = i["resolution"]["height"], i["resolution"][
                        "width"]
                    aspect = self.max_height / self.max_width
                    hw_aspect_thr = 1.5
                    is_pick = filter_resolution(
                        height,
                        width,
                        max_h_div_w_ratio=hw_aspect_thr * aspect,
                        min_h_div_w_ratio=1 / hw_aspect_thr * aspect,
                    )
                    if not is_pick:
                        print("resolution mismatch")
                        cnt_resolution_mismatch += 1
                        continue

                # import ipdb;ipdb.set_trace()
                i["num_frames"] = math.ceil(fps * duration)
                # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration.
                if i["num_frames"] / fps > self.video_length_tolerance_range * (
                        self.num_frames / self.train_fps * self.speed_factor
                ):  # too long video is not suitable for this training stage (self.num_frames)
                    cnt_too_long += 1
                    continue

                # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                frame_interval = fps / self.train_fps
                start_frame_idx = 0
                frame_indices = np.arange(start_frame_idx, i["num_frames"],
                                          frame_interval).astype(int)

                # comment out it to enable dynamic frames training
                if (len(frame_indices) < self.num_frames
                        and random.random() < self.drop_short_ratio):
                    cnt_too_short += 1
                    continue

                #  too long video will be temporal-crop randomly
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(
                        len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                    # frame_indices = frame_indices[:self.num_frames]  # head crop
                i["sample_frame_index"] = frame_indices.tolist()
                new_cap_list.append(i)
                i["sample_num_frames"] = len(
                    i["sample_frame_index"]
                )  # will use in dataloader(group sampler)
                sample_num_frames.append(i["sample_num_frames"])
            elif path.endswith(".jpg"):  # image
                cnt_img += 1
                new_cap_list.append(i)
                i["sample_num_frames"] = 1
                sample_num_frames.append(i["sample_num_frames"])
            else:
                raise NameError(
                    f"Unknown file extension {path.split('.')[-1]}, only support .mp4 for video and .jpg for image"
                )
        # import ipdb;ipdb.set_trace()
        main_print(
            f"no_cap: {cnt_no_cap}, too_long: {cnt_too_long}, too_short: {cnt_too_short}, "
            f"no_resolution: {cnt_no_resolution}, resolution_mismatch: {cnt_resolution_mismatch}, "
            f"Counter(sample_num_frames): {Counter(sample_num_frames)}, cnt_movie: {cnt_movie}, cnt_img: {cnt_img}, "
            f"before filter: {len(cap_list)}, after filter: {len(new_cap_list)}"
        )
        return new_cap_list, sample_num_frames

    def decord_read(self, path, frame_indices):
        decord_vr = self.v_decoder(path)
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1,
                                        2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def read_jsons(self, data):
        cap_lists = []
        with open(data, "r") as f:
            folder_anno = [
                i.strip().split(",") for i in f.readlines()
                if len(i.strip()) > 0
            ]
        print(folder_anno)
        for folder, anno in folder_anno:
            with open(anno, "r") as f:
                sub_list = json.load(f)
            for i in range(len(sub_list)):
                sub_list[i]["path"] = opj(folder, sub_list[i]["path"])
            cap_lists += sub_list
        return cap_lists

    def get_cap_list(self):
        cap_lists = self.read_jsons(self.data)
        return cap_lists


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

class WANVideoDataset(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 832,
        video_reshape_mode: str = "center",
        fps: int = 8,
        max_num_frames: int = 8,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()
        import json
        with open('dataset_meta.json', 'r', encoding='utf-8') as file:
            self.data = json.load(file)
            print(len(self.data))
        self._client =  Client('~/petreloss.conf', enable_mc=True)
        self.klist = list(self.data.keys())
        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""
        self.vae = AutoencoderKLWan.from_pretrained("pretrained/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("pretrained/Wan2.1-T2V-1.3B-Diffusers", subfolder="tokenizer", torch_dtype=torch.bfloat16)
        self.text_encoder = UMT5EncoderModel.from_pretrained("pretrained/Wan2.1-T2V-1.3B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16).cuda()

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.uncond_prompt_embed, self.uncond_prompt_mask = self._get_t5_prompt_embeds(
                    prompt=[""],
                    device=torch.device('cuda'),
                )
                self.uncond_prompt_embed, self.uncond_prompt_mask =self.uncond_prompt_embed.squeeze(0), self.uncond_prompt_mask.squeeze(0).bool()
        self.cfg_rate = 0.1

    def __len__(self):
        return len(self.klist)
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        te_output = self.text_encoder(text_input_ids.to(device), mask.to(device))
        prompt_embeds = te_output.last_hidden_state
        attention_mask = mask.bool().to(device)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, attention_mask


    def __getitem__(self, index):
        item = self.data[self.klist[index]]
        vid_path = item['video']
        video_bytes = self._client.get(vid_path)#,update_cache=True)
        video_bytes = io.BytesIO(video_bytes)
        while video_bytes.getbuffer().nbytes == 0:
            index+=1
            item = self.data[self.klist[index]]
            vid_path = item['video']
            video_bytes = self._client.get(vid_path)#,update_cache=True)
            video_bytes = io.BytesIO(video_bytes)
        
        vr = decord.VideoReader(video_bytes)

        videos = self._preprocess_data(vr)[0].unsqueeze(0).permute((0,2,1,3,4))

        while videos.shape[2]!=self.max_num_frames or videos.shape[3]!=480 or videos.shape[4]!=832:
            index += 1
            item = self.data[self.klist[index]]
            vid_path = item['video']
            video_bytes = self._client.get(vid_path)#,update_cache=True)
            video_bytes = io.BytesIO(video_bytes)
            while video_bytes.getbuffer().nbytes == 0:
                index+=1
                item = self.data[self.klist[index]]
                vid_path = item['video']
                video_bytes = self._client.get(vid_path)#,update_cache=True)
                video_bytes = io.BytesIO(video_bytes)
            
            vr = decord.VideoReader(video_bytes)

            videos = self._preprocess_data(vr)[0].unsqueeze(0).permute((0,2,1,3,4))

        prompt = item['text'] +' '+item['caption']
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = self.vae.encode(videos.to(self.vae.device))["latent_dist"].sample()
                prompt_embeds, attention_mask = self._get_t5_prompt_embeds(
                prompt=[prompt],
                device=torch.device('cuda'),
                )
        
        if random.random() < self.cfg_rate:
            prompt_embeds = self.uncond_prompt_embed
            attention_mask = self.uncond_prompt_mask
            return latents.squeeze(0).cpu(), prompt_embeds.squeeze(0).cpu(), attention_mask.squeeze(0).cpu()
        else:
            return latents.squeeze(0).cpu(), prompt_embeds.squeeze(0).cpu(), attention_mask.squeeze(0).cpu()
    
    def latent_collate_function(batch):
        # return latent, prompt, latent_attn_mask, text_attn_mask
        # latent_attn_mask: # b t h w
        # text_attn_mask: b 1 l
        # needs to check if the latent/prompt' size and apply padding & attn mask
        latents, prompt_embeds, prompt_attention_masks = zip(*batch)
        # calculate max shape
        max_t = max([latent.shape[1] for latent in latents])
        max_h = max([latent.shape[2] for latent in latents])
        max_w = max([latent.shape[3] for latent in latents])

        # padding
        latents = [
            torch.nn.functional.pad(
                latent,
                (
                    0,
                    max_t - latent.shape[1],
                    0,
                    max_h - latent.shape[2],
                    0,
                    max_w - latent.shape[3],
                ),
            ) for latent in latents
        ]
        # attn mask
        latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)
        # set to 0 if padding
        for i, latent in enumerate(latents):
            latent_attn_mask[i, latent.shape[1]:, :, :] = 0
            latent_attn_mask[i, :, latent.shape[2]:, :] = 0
            latent_attn_mask[i, :, :, latent.shape[3]:] = 0

        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
        latents = torch.stack(latents, dim=0)
        return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def _resize_for_rectangle_crop(self, arr):
        image_size = self.height, self.width
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_data(self,vr):
        decord.bridge.set_bridge("torch")
        videos = []

        # for filename in self.instance_video_paths:
        if True:
            video_reader = vr
            video_num_frames = len(video_reader)

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                frames = video_reader.get_batch(indices)

            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0

            # Training transforms
            frames = (frames - 127.5) / 127.5
            frames = frames.permute(0, 3, 1, 2)  # [F, C, H, W]

            frames = self._resize_for_rectangle_crop(frames)
            videos.append(frames.contiguous())  # [F, C, H, W]
        return videos