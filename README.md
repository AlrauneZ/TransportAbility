[![GS-Frame](https://img.shields.io/badge/github-GeoStat_Framework-468a88?logo=github&style=flat)](https://github.com/GeoStat-Framework)
[![Gitter](https://badges.gitter.im/GeoStat-Examples/community.svg)](https://gitter.im/GeoStat-Examples/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# Template

This is a template for an example repository.

You can create a new example by simply clicking on "Use this template".

The included example is showing the generation of a conditioned random field ensemble
in 1D taken from [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/06_conditioned_fields/00_condition_ensemble.html#sphx-glr-examples-06-conditioned-fields-00-condition-ensemble-py).


## Structure

Please try to organize your example in the given Structure
- `data/` - here you should place your input data
- `src/` - here you should place your python scripts
- `results/` - here your computed results and plots should be stored
- `README.md` - please describe your example in the readme, potentially showing results
- `LICENSE` - the default license is MIT, you can use another one if wanted


## Python environment

To make the example reproducible, it would be a good practice to provide one of
the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages
- `spec-file.txt` - specification file to create the original [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)


## Workflow

After finalizing your work, you should tag the repository with a version like `v1.0`.

Then, a [Zenodo](https://zenodo.org/) release will be created, so you can cite the repository in you publication.

Please keep your `master` branch in line with the latest release.
For further development use the `develop` branch and update `master` with pull-requests.


## Contact

You can contact us via <info@geostat-framework.org>.


## License

MIT © 2020
