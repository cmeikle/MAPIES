from mapies.viirs import VIIRS


if __name__ == "__main__":
    # Change these variables
    start_date = "202401010830"
    end_date = "202401011630"
    outdir="/home/cmeikle/Projects/data/"
    indir="/home/cmeikle/Projects/data/VIIRS/original_files/AERDB_L2_VIIRS_NOAA20"
    c = VIIRS(start_date, end_date, frequency="3H", dest=outdir, indir=indir)
    c.read_nc()
    c.preprocess_vars()

    c.plot_2D_obs()
    #c.to_da()