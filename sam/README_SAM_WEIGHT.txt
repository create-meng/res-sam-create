SAM checkpoint layout (Res-SAM fork)

- 本仓库主线与作者公开仓库一致：SAM ViT-L，权重文件名为 sam/sam_vit_l_0b3195.pth（约 1.2GB），与 sam/sam.py 同目录（仓库根下 sam/ 为克隆自带，勿放到其它路径）。
- GitHub 远程可通过 Git LFS 托管上述完整文件；OpenI 等平台往往无法上传或拉取如此大的 LFS 对象，请克隆后从 Meta 官方地址手动下载到 sam/（文件名需一致）。推荐使用浅克隆：`git clone --depth 1`（配合 `GIT_LFS_SKIP_SMUDGE=1` 见仓库 README）。
- PatchRes/sam_integration.py 中 SamAutomaticMaskGenerator 的超参与作者仓库 sam/sam.py 一致，与 vit_l / vit_b / vit_h 作为骨干独立（需与 checkpoint 匹配）。

Official ViT-L URL:
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

推荐 aria2 多线程下载（与 setup_after_clone_*.sh 中命令一致）：
  apt-get update && apt-get install -y aria2
  aria2c -x 16 -s 16 -k 1M -o sam_vit_l_0b3195.pth -d sam "<上列 URL>"
