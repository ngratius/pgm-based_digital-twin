# PGM-based digital-twin

Lessons learned on the implementation of probabilistic graphical model-based digital twins: A space habitat study

Related publication: [https://doi.org/10.1016/j.jsse.2023.04.001](https://doi.org/10.1016/j.jsse.2023.04.001)

## Description

Habitats for future human spaceflights will require more resilient environmental control and life support systems (ECLSS). To that end, it is important to facilitate decision making in case of unexpected failure by quantifying the uncertain and dynamic nature of the physical phenomena involved. Combining probabilistic and deterministic models is a particularly promising approach to address this issue. In particular, Probabilistic Graphical Model (PGM) based digital twins are relevant as they embed random variables evolving overtime. Previous research used this modeling method for several applications such as monitoring structural health or manufacturing processes. We envision that the space exploration sector can also benefit from this approach by using the insight gained on specific sub-systems. In this study, we propose lessons learned on the implementation process of PGM-based Digital Twin to quantify uncertainties for temperature prognosis in ECLSS. These findings are introduced as a step-by-step guideline which result in developing a probabilistic model applicable to space habitats. We focused on directed acyclic graphs as this type of PGM can integrate expert's knowledge with data which has been proven to enhance accuracy. A literature review was conducted to identify the state-of-the-art practices and the proposed lessons learned were derived from the study of a physical infrastructure meant to predict the behavior of a space habitat. A temperature control failure scenario was considered, and the Digital Twin estimated the time available before the temperature would become critical. Experiments were conducted on three office rooms to simulate the behavior of an ECLSS. The model was trained offline using historical sensor data and performed inference online by computing the conditional probability of a multivariate normal density. We found that a successful implementation process requires to iteratively go through four stages: outline, design, calibrate and evaluate. It involves selecting ECLSS-relevant functionalities and an associated decision-making problem that relies on habitability criteria. Observable variables must be chosen according to a sensors architecture that is compatible with a typical habitat infrastructure. As real space systems are not easily available for model validation, we suggest evaluating early designs on high-fidelity analogs. In future work, we envisage to further assess the impact of the design stage on the model's performance by considering computational cost and inference capability.

## Dependencies

This project relies on the following Python packages:

- pandas
- numpy
- matplotlib
- scipy

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib scipy
```

## Authors

* [Nicolas Gratius](https://www.linkedin.com/in/nicolas-gratius-3360b0110/)

* [Mario Bergés](https://www.cmu.edu/cee/people/faculty/berges.html)

* [Burcu Akinci](https://www.cmu.edu/cee/people/faculty/akinci.html)

## Acknowledgments

[NASA grant 80NSSC19K1052](https://govtribe.com/award/federal-grant-award/grant-for-research-80nssc19k1052)
