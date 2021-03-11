# ICLR Computational Geometry & Topology Challenge 2021

[![Build Status](https://travis-ci.org/geomstats/challenge-iclr-2021.svg?branch=main)](https://travis-ci.org/geomstats/challenge-iclr-2021) 

Welcome to the ICLR Computational Geometry & Topology challenge 2021!

The purpose of this challenge is to push forward the fields of computational differential geometry and topology, by creating the best data analysis, computational method, or numerical experiment relying on state-of-the-art geometric and topological Python packages.

All participants will have the opportunity to co-author a white paper summarizing the findings of the challenge.

Each submission takes the form of a Jupyter Notebook leveraging the packages [Geomstats](https://github.com/geomstats/geomstats) and [Giotto-tda](https://github.com/giotto-ai/giotto-tda). The participants submit their Jupyter Notebook via [Pull Requests](https://github.com/geomstats/challenge-iclr-2021/pulls) (PR) to this GitHub repository, see [Guidelines](#guidelines) below.

## Deadline

The final Pull Request submission date and hour will have to take place before:
- **May 2nd, 2021 at 23:59 PST (Pacific Standard Time)**. 

The participants can freely commit to their Pull Request and modify their submission until this time.

## Winners announcement and prizes

The first 3 winners will be announced at the ICLR 2021 virtual workshop [Geometrical and Topological Representation Learning](https://gt-rl.github.io/) and advertised through the web. The winners will also be contacted directly via email. 

The prizes are:
- 2000$ for the 1st place,
- 1000$ for the 2nd place,
- 500$ for the 3rd place.
 
## Subscription

Anyone can participate and participation is free. It is enough to:
- send a [Pull Request](https://github.com/geomstats/challenge-iclr-2021/pulls),
- accept the Contributor License Agreement (CLA) on GitHub,
- follow the challenge [guidelines](#guidelines),
to be automatically considered amongst the participants. 

An acceptable PR automatically subscribes a participant to the challenge.

## Guidelines

We encourage the participants to start submitting their Pull Request early on. This allows to debug the tests and helps to address potential issues with the code.

A submission should respect the following Jupyter Notebook’s structure:
- Introduction and motivation: 
    - Explain and motivate the problem
- Analysis/Experiment:
    - Describe the dataset (if your submission analyzes on a dataset)
    - Detail and motivate the steps of the analysis/experiment
    - Describe and highlight the role of Geomstats and Giotto-tda
- Benchmark:
    - Compare your results with other methods/packages
- Limitations and perspectives:
    - Describe the limitations of your analysis/method/experiment
    - Describe the limitations of the packages Geomstats and Giotto-tda
    - List features that you would like to find in packages like Geomstats and Giotto-tda

Here is a non-exhaustive list of possible submissions:
- Data analysis with geometric and topological methods,
- Implementation of the code from a research paper with Geomstats and Giotto-tda
- Implementation of a feature to merge into Geomstats and Giotto-tda codebases (see examples of PR on Geomstats and Giotto-tda repositories)
- Implementation of a visualization method to merge into Geomstats and Giotto-tda (see example here: )
- Benchmarking/profiling on geometric and topological methods against other methods for a public dataset.
Etc.

The notebooks provided in the `submission-example-*` folders are examples of data analysis submissions that can help the participants to design their proposal and to understand how to use the packages. Note that these examples are "naive" on purpose and are only meant to give illustrative notebooks templates rather than to provide a meaningful data analysis. More examples on how to use the packages can be found on the GitHub repositories of [Geomstats](https://github.com/geomstats/geomstats) and [Giotto-tda](https://github.com/giotto-ai/giotto-tda).

The code should be compatible with Python 3.8 and make an effort to respect the Python style guide [PEP8](https://www.python.org/dev/peps/pep-0008/). The portion of the code using `geomstats` only needs to run with `numpy` backend, `pytorch` and `tensorflow` backends are not required.

The Jupyter notebooks are automatically tested when a Pull Request is submitted. The tests have to pass. Their running time should not exceed 3 hours, although exceptions can be made by contacting the challenge organizers.

If a dataset is used, the dataset has to be public and referenced. There is no constraint on the data type to be used.

A participant can raise GitHub issues and/or request help or guidance at any time through [Geomstats slack](https://geomstats.slack.com/) and [Giotto-TDA slack](https://slack.giotto.ai/). The help/guidance will be provided modulo availability of the maintainers.

**Important:** Geomstats *and* Giotto-tda have to play a central role in the submission.


## Submission procedure

1. Fork this repository to your GitHub.

2. Create a new folder with your team leader's GitHub username in the root folder of the forked repository, in the main branch.

3. Place your submission inside the folder created at step 2, with:
- the unique Jupyter notebook (the file shall end with .ipynb),
- datasets (if needed),
- auxiliary Python files (if needed).

Datasets larger than 10MB shall be directly imported from external URLs or from data sharing platforms such as OpenML.

If your project requires external pip installable libraries that are not amongst Geomstats’ and Giotto-tda’s requirements.txt, you can include them at the beginning of your Jupyter notebook, e.g. with:
```
import sys
!{sys.executable} -m pip install numpy scipy torch
```

## Evaluation and ranking

The [Condorcet method](https://en.wikipedia.org/wiki/Condorcet_method) will be used to rank the submissions and decide on the winners. 

Selected Geomstats/Giotto-tda maintainers and collaborators, as well as each participant whose submission respects the guidelines, will vote once on Google Form to express their preference for the 3 best submissions. The 3 preferences must all 3 be different: e.g. one cannot select the same Jupyter notebook for both first and second place. Such irregular votes will be discarded. A link to a Google Form will be provided to record the votes. It will be required to insert an email address to identify the voter. The voters will remain secret, only the final ranking will be published.

## Questions?

Feel free to contact us through [GitHub issues on this repository](https://github.com/geomstats/challenge-iclr-2021/issues), on Geomstats/Giotto-TDA repositories or through [Geomstats slack](https://geomstats.slack.com/) and [Giotto-TDA slack](https://slack.giotto.ai/).
