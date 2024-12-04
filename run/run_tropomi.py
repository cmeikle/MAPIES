from mapies.tropomi import TROPOMI


if __name__ == "__main__":
    start_date = "202301010000"
    end_date = "202301012359"
    outdir="/home/cmeikle/Projects/data/TROPOMI"
    indir="/home/cmeikle/Projects/data/TROPOMI"
    c = TROPOMI(start_date, end_date, indir=indir, dest=outdir)
    c.read_nc()
    c.preprocess_vars()
    c.plot_2D_obs()