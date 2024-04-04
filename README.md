# substance_abuse_treatment_analysis

This project aims to identify factors that contribute to the failure of treating substance abuse at the state level in the United States. By employing exploratory data analysis and machine learning techniques, including Support Vector Machines (SVMs) and Neural Networks (NNs), we uncover patterns between variables and treatment outcomes for different states. The analysis reveals that the type of substance and treatment service are the most significant variables in determining the outcome of treatment.

This project was developed for the Merck Datathon by Team 19.



## Key Findings
States such as Kentucky, North Carolina, and Hawaii face more serious challenges in treating substance abuse.
The SVM models highlight the influence of different features on treatment outcomes through the magnitude of the linear coefficients.
The NN model captures complex non-linear relationships, which can be visualized by Shapley values.
Length of stay (LOS) is identified as the most important feature for treatment success, indicating that longer treatment sessions can result in better outcomes.

More findings and detailed analysis can be found in the [report](./Team_19_report.pdf).

## Interactive Dashboard
**Click on the image** to explore the geographic distribution of treatment facilities and failure rates across the US through our Interactive Tableau Dashboard.

[![Interactive Dashboard](./data_viz/tableau_preview.png)](https://public.tableau.com/app/profile/hanyu.liu8152/viz/SubstanceAbusementtTreatmentAcrosstheUS/FacilityDistribution)

## Data Sources
The analysis is based on treatment records, additional datasets providing information on state and statistical areas, and the American Community Survey (ACS) data.
Source data can be found in the [data folder](./data) 

## Modeling Results
Our analysis employed machine learning techniques to predict treatment outcomes and identify significant factors contributing to treatment failure. Here are some key results from our modeling:

- Support Vector Machines (SVMs): The SVM models demonstrated the influence of different features on treatment outcomes through the magnitude of the linear coefficients. The best-performing SVM model achieved an accuracy of 76.4% on the test set.

- Neural Networks (NNs): The NN model captured complex non-linear relationships, which were visualized using Shapley values. The neural network achieved a higher prediction accuracy of 84.8% on the test set, indicating its effectiveness in capturing non-linear patterns.

- Feature Importance: The length of stay (LOS) was identified as the most important feature for treatment success, suggesting that longer treatment sessions can result in better outcomes. Other significant features included the type of substance, treatment service, and patient demographics.

<img src="./data_viz/shapley%20values.png" alt="Alt text" width="400"/>



## Conclusion
To improve the quality of substance abuse treatment, state governments and healthcare officials should focus on orchestrating substance-specific programs that cater to the needs of patients. Early intervention, medication assistance, ensuring accessibility of programs, decriminalization, and controlling the cost of healthcare are potential solutions.

