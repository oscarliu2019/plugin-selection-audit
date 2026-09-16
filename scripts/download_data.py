"""下载长期预测数据集并核验行数/通道数。

数据源：TSLib 作者镜像在 HuggingFace 的 `thuml/Time-Series-Library`（dataset 仓库），
可脚本化下载，避免 Google Drive / 清华云盘的人工点击。8 个主数据集约 123 MB，
加上可选的 Traffic 约 265 MB。

用法
----
    python scripts/download_data.py                    # 下载 8 个主数据集
    python scripts/download_data.py --include-optional # 追加 Traffic（862 通道，130 MB+）
    python scripts/download_data.py --datasets ETTh1 ILI
    python scripts/download_data.py --verify-only      # 只核验已下载文件

核验逻辑：读 CSV 的行数与列数，与 configs/matrix.yaml 里声明的 total_len / enc_in 比对。
不一致会直接报错——这是防止「下到错版本数据导致整套结果不可比」的第一道闸门。
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, MatrixConfig  # noqa: E402


def _download_hf_hub(repo_id: str, hf_path: str, dest: Path) -> bool:
    """优先走 huggingface_hub（支持断点/缓存/镜像 HF_ENDPOINT）。"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        p = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=hf_path)
    except Exception as exc:  # 网络/权限问题时回退到直链
        print(f"    hf_hub 失败（{type(exc).__name__}: {exc}），回退直链")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(p).read_bytes())
    return True


def _download_url(base_url: str, hf_path: str, dest: Path) -> None:
    url = f"{base_url}/{hf_path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def verify(cfg: MatrixConfig, name: str) -> tuple[bool, str]:
    """核验行数与通道数是否与 matrix.yaml 声明一致。"""
    ds = cfg.dataset(name)
    csv = REPO_ROOT / ds["root_path"] / ds["data_path"]
    if not csv.exists():
        return False, "文件不存在"
    df = pd.read_csv(csv)
    n_rows = len(df)
    n_ch = len([c for c in df.columns if c.lower() != "date"])
    ok = n_rows == int(ds["total_len"]) and n_ch == int(ds["enc_in"])
    msg = (f"rows={n_rows}(期望 {ds['total_len']}) channels={n_ch}(期望 {ds['enc_in']}) "
           f"size={csv.stat().st_size/1e6:.1f}MB")
    return ok, msg


def main() -> None:
    ap = argparse.ArgumentParser(description="下载并核验长期预测数据集")
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--include-optional", action="store_true", help="包含 Traffic")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    cfg = MatrixConfig.load(args.config)
    src = cfg.raw["data_source"]
    names = args.datasets or cfg.dataset_names(include_optional=args.include_optional)

    failed: list[str] = []
    for name in names:
        ds = cfg.dataset(name)
        dest = REPO_ROOT / ds["root_path"] / ds["data_path"]
        if not args.verify_only and (args.force or not dest.exists()):
            print(f"[data] 下载 {name} -> {dest.relative_to(REPO_ROOT)}")
            if not _download_hf_hub(src["hf_repo_id"], ds["hf_path"], dest):
                _download_url(src["base_url"], ds["hf_path"], dest)
        ok, msg = verify(cfg, name)
        print(f"[data] {'OK  ' if ok else 'FAIL'} {name:12s} {msg}")
        if not ok:
            failed.append(name)

    print(f"\n[data] {len(names) - len(failed)}/{len(names)} 个数据集通过核验")
    if failed:
        print(f"[data] 未通过: {failed}（重跑本脚本或加 --force 重下）")
        sys.exit(1)


if __name__ == "__main__":
    main()
