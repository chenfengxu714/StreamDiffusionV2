import json
import sys
import subprocess
import os


commands = []
os.makedirs("outputs", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

with open("../evaluation.json", 'r') as f:
    tests = json.load(f)

for test in tests:
    video_name = test['src_video_name']
    height = test['height']
    width = test['width']
    fps = test['fps']
    prompt = test['prompt']
    output_name = test['output_video_name']

    with open(f"prompts/{output_name}.txt", 'w') as f:
        f.write(prompt)

    for name, ckpt_folder, config_file, fixed_noise_scale in [
        ("default", "wan_causal_dmd_v2v", "wan_causal_dmd_v2v.yaml", False),
        ("kv_cache_21", "wan_causal_dmd_v2v", "wan_causal_dmd_v2v_kv_cache_21.yaml", False),
        ("sink_token_0", "wan_causal_dmd_v2v", "wan_causal_dmd_v2v_sink_tokens_0.yaml", False),
        ("fixed_noise_scale", "wan_causal_dmd_v2v", "wan_causal_dmd_v2v.yaml", True),
        ("causvid", "autoregressive_checkpoint", "wan_causal_dmd_v2v_causvid.yaml", True),
    ]:

        command = f'''\
python streamv2v/inference.py \
--config_path configs/{config_file} \
--checkpoint_folder ckpts/{ckpt_folder} \
--output_folder outputs/{name}/{output_name} \
--prompt_file_path prompts/{output_name}.txt \
--video_path ../videos/{video_name}.mp4 \
--height {height} \
--width {width} \
--fps {fps} \
--step 2\
'''
        if fixed_noise_scale:
            command += " --fixed_noise_scale"
        commands.append(command)
        print(command)

print("Start")
sys.stdout.flush()

for cmd in commands:
    print(cmd)
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stderr:
            print("Command error output:", result.stderr)
        if result.stdout:
            print("Command output:")
            print(result.stdout)
        sys.stdout.flush()
    except subprocess.CalledProcessError as e:
        print("\n--- Command failed with error ---")
        print(f"Error: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Command run: {e.cmd}")

        if e.stdout:
            print("\n--- Captured stdout ---")
            print(e.stdout)

        if e.stderr:
            print("\n--- Captured stderr ---")
            print(e.stderr)
        sys.stdout.flush()

        break

print("Finished")
sys.stdout.flush()
