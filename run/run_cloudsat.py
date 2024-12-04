from mapies.cloudsat import CLOUDSAT
from dask.distributed import Client



if __name__ == "__main__":

    # Change these variables
    start_date = "202001010000"
    end_date = "202001310000"
    outdir="/home/cmeikle/Projects/data/CLOUDSAT"
    indir="/home/cmeikle/Projects/data/CLOUDSAT/original_files"
    c = CLOUDSAT(start_date, end_date, indir=indir, dest=outdir)
    c.read_multiple_files()
    c.plot_2D_obs()