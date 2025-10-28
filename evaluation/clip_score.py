from PIL import Image
import json 
import os 
import torch 
import cv2
import sys

from transformers import CLIPProcessor, CLIPModel


device = "cuda"

# Function to create a dictionary with vid_name as keys and prompt as values
def create_vid_prompt_dict(json_data):
    vid_prompt_dict = {}
    for item in json_data:
        vid_name = item["output_video_name"]
        prompt = item["prompt"]
        vid_prompt_dict[vid_name] = prompt
    return vid_prompt_dict

if __name__ == "__main__":
    os.makedirs("clip_scores", exist_ok=True)

    method_names = [
        "user_study_caus_vid",
        "user_study_stream_diffusion",
        "user_study_stream_diffusion_v2",
        "ablation_noise_scale_0.8",
        "ablation_num_kv_cache_21",
        "ablation_num_sink_tokens_0",
    ]

    with open("evaluation.json", 'r') as file:
        json_data = json.load(file)
    video_maps = create_vid_prompt_dict(json_data)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for method_name in method_names:
        print(f"Evaluating {method_name}")
        edit_video_dir = f"{method_name}"
        video_names = list(video_maps.keys())

        consistency_score = []
        prompt_score = []

        out_json = {}

        for v in video_names:
            try:
                out_json[v] = {}
                prompt = video_maps[v]
                video_path = os.path.join(edit_video_dir, f"{v}.mp4")
                video_embs = []

                # Open the video file
                cap = cv2.VideoCapture(video_path)

                # Check if video opened successfully
                if not cap.isOpened():
                    print("Error opening video file")
                # Process video frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Convert the BGR frame captured by cv2 to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Convert the numpy array frame to PIL Image
                        image = Image.fromarray(frame_rgb)
                        
                        # Your existing processing code
                        with torch.no_grad():
                            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            outputs = model(**inputs)

                            image_embeds = outputs.image_embeds
                            text_embeds = outputs.text_embeds
                        video_embs.append(image_embeds)
                    else:
                        # Break the loop if no frames are returned (end of video)
                        break

                # Release the video capture object
                cap.release()
                video_embs = torch.cat(video_embs, dim=0)   # (T, 768)

                text_score = cos(text_embeds, video_embs)    # (1, T)
                text_score = text_score.mean().cpu().item()
                prompt_score.append(text_score)

                # two continue frames cos similarity
                emb1 = video_embs[:-1]  # (N, 768)
                emb2 = video_embs[1:]   # (N, 768)
                score = cos(emb1, emb2) # (N,)
                score = score.mean().cpu().item()

                consistency_score.append(score)
                out_json[v]["prompt"] = prompt
                out_json[v]["consistency_score"] = score
                out_json[v]["prompt_score"] = text_score
                print(v, prompt, score)
            except:
                print(f'{v} has error!')
            sys.stdout.flush()

        print("Number of videos ", len(prompt_score))
        print("Avg consistency score ", sum(consistency_score) / len(consistency_score))
        print("Avg prompt score ", sum(prompt_score) / len(prompt_score))

        json.dump(out_json, open(f"clip_scores/{method_name}.json", "w"), sort_keys=True, indent=2)
