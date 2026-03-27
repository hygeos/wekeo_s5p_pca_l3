from wekeo_s5p_pca_l3.hygeos_core import env
output_dir = env.getdir("OUTPUT_DIR")
frp_download_dir = env.getdir("DIR_ANCILLARY") / "SLSTR_FRP"

if not frp_download_dir.exists():
    raise FileNotFoundError(f"FRP download directory {frp_download_dir} does not exist. Please create it or check your environment configuration.")

if not output_dir.exists():
    raise FileNotFoundError(f"Output directory {output_dir} does not exist. Please create it or check your environment configuration.")


gridded_s5p_pca_dir = output_dir / "gridded_s5p_pca_l3"
gridded_s5p_pca_dir.mkdir(parents=False, exist_ok=True)