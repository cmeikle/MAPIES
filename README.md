MAPIES
Monarch Assimilation Preproc and Inversion of Emissions plus Satellites

To setup on your own machine, I would recommend creating a virtual environment t. For example,

```
python -m env venv
```

This will set up a virtual environment. From there you can install all python modules you want by running

```
pip install -r requirements.txt
```

Next to run the tool, we have examples in the `mapies/run` folder. For example you can run the run_viirs.py file with a simple `python run_viirs.py` after updating the necessary variables. The dates and the directories. 

Presently you will need to run `export PYTHONPATH=../mapies` from inside this directory for the tool to see the in-built modules. We are working on a way to incorporate this into the pipeline.

Happy mapping!



