import os
# 注意os.environ得在import huggingface库相关语句之前执行。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
def download_model(local_dir,repo_id,token):
    # print(f'开始下载\n仓库：{repo_id}\n大模型：{filename}\n如超时不用管，会自定继续下载，直至完成。中途中断，再次运行将继续下载。')
    while True:   
        try:
            snapshot_download(local_dir=local_dir,
            repo_id=repo_id,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=100
            )
        except Exception as e :
            print(e)
        else:
            # print(f'下载完成，大模型保存在：{local_dir}\{filename}')
            break
            
if __name__ == '__main__':
    repo_id = 'Qwen/Qwen3-VL-4B-Instruct'
    token = os.environ.get('HF_TOKEN')  # set HF_TOKEN in your shell, e.g. `export HF_TOKEN=hf_...`
    if not token:
        raise SystemExit("HF_TOKEN env var is not set. Get a token at https://huggingface.co/settings/tokens")
    local_dir = './playground/Pretrained_models/Qwen3-VL-4B-Instruct'
    download_model(local_dir, repo_id, token)
