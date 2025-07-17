## Group 3 Project Proposal

Seng 404

Richard Gage V00801959

Jack Hamilton V00985614

Santiago Velasco V01050218


## The Software Problem

Maintaining sustainable project management in collaborative environments such as GitHub is a very important problem in software development. 

Inspired by Jorge Aranda and Gina Venolia's paper, "*The Secret Life of Bugs: Going Past the Errors and Omissions in Software Repositories*," this project aims to study the early life of bugs. We will explore the relationship between the number of open issues in repositories and other characteristics, such as the number of contributors, the programming languages used, and the project descriptions.

Understanding what predicts open issue buildup can help project leaders manage sustainability, make better informed decisions about optimal team size and technology choices, ultimately increasing the chance of a project's success. We found that most existing research focuses on co-located development teams, and so we wanted to focus on distributed projects on platforms like GitHub.


## Research Question & Hypotheses

The primary research question we want to answer is: **What parameters in a repository can be used to predict the number of open issues the project has?**

Our paper will contribute to the overall health and sustainability of future projects. This is an important matter, as open issues in high numbers can reflect poor maintenance or an overwhelming demand that the contributors cannot keep up with. Understanding this relation can help contributors gauge the health of their projects, and even the scalability depending on the number of contributors available, the language being used, or the difficulty of the project at hand.


### Hypotheses

Based on our preliminary data analysis we propose the following testable hypotheses:


#### H1: Positive Contributor to Issue Relationship

Repositories with more contributors will have significantly more open issues - a Pearson correlation coefficient r > 0.3.


#### H2: Programming Language Effect

Certain programming languages will be significant predictors of open issue counts. We expect repositories using Python and Javascript to have more open issues than those using lower-level languages such as C/C++ due to larger user bases and faster development cycles.


### Research Strategies

Our leading research strategies are to analyze potential correlations within the data—partly in our effort to check if the relationship is truly linear—as well as statistical visualizations such as to help understand the distribution of contributors. Regression techniques can be used as well to help us understand the strength of these relationships. Through this investigation, we aim to identify correlations between these parameters and determine potential outcomes. 

Looking towards related work, "The Secret Life of Bugs: Going Past the Errors and Omissions in Software Repositories" highlights how repositories often omit key elements of the software development process, including communication, rationale, and personal dynamics. Our work builds upon this to further understand and uncover these aspects of bug creations, as well as their resolution, through patterns within repository statistics.


## Methodology


### Our Primary Dataset for Pilot Experiments

We found a dataset containing data from 25,999,958 repositories in the Google Cloud marketplace. We do not expect to use the entire dataset, only a sample of 5,000. The following plots and information were obtained by conducting some initial experiments with the reduced dataset. The complete code used can be found on the GitHub repository.

To start, a five number summary from the data distribution was computed:


```
count    5000.000000
mean       11.779400
std        71.733026
min         1.000000
25%         1.000000
50%         2.000000
75%         7.000000
max      3317.000000
```


Table 01. Five number summary and quartiles of the data.

This shows that the data is right-skewed and may contain large outliers. After removing outliers and large values, we used a violin plot, a box plot, and a bar chart together to better understand how the data is distributed. A total of 223 samples were removed from the data used to plot the following graphics because they contained extreme values.



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


Figure 01. Violin and Box plot of the Number of Open Issues (Maximum of 40 open issues).



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")


Figure 02. Bar Chart of the Frequency of the Number of Open Issues.

Finally, a scatter plot was created to compare the number of open issues to the number of contributors for each repository.



<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.png "image_tooltip")


Figure 03. Number of Open Issues vs Number of Contributors

As shown in the previous figure, there does not appear to be a correlation between the two variables; rather, the data points seem to be spread out randomly. However, there is a significant concentration of points in the lower left corner.


### Projected WorkFlow


#### Phase 1: Data Preparation



1. Data Source
2. Sampling
3. Filtering Criteria
4. Feature Extraction
5. Data Cleaning

#### 
    Phase 2: Statistical Analysis

1. Exploratory data analysis
2. Feature correlation analysis
3. Regression model building
4. Model validation and testing
5. Results interpretation


#### Working Tools

The tools needed for this project will help us process, clean, visualize, and retrieve statistics from the data. Although R is a good option for analysis, we would like to take a different approach. We plan to use Python with specialized libraries, such as NumPy and Pandas, to store and manipulate large arrays containing the data we are working with. We will also use SciPy to perform test statistics and Matplotlib to create plots and charts.


## Expected Results

Based on our primary data analysis and our hypothesis, we expect to find a positive correlation between the number of contributors and open issues. Our initial scatter plot suggests that this relationship might be non-linear, with high variability on smaller projects. We also expect notable differences across different programming languages, with web focused languages having the highest issue counts, and systems languages being the lowest.

This research will provide:



* empirical validation to the relationship between team size and project outcomes
* a framework for analyzing open source repository health
* baseline metrics for future studies on software project sustainability


## Limitations and Threads to Validity

**Potential Limitations and Threads to Validity Include:**



* Missing variables
* Given that most of our data will be drawn from correlation, it will be difficult to prove its causation
* Outlier Sensitivity
* We don’t have time as a variable, meaning that some of these repositories may be littered with issues made in recent activity, which doesn’t necessarily reflect contributor count
* Selection Bias


## Project Timeline


## References


    Berger, E. D., Hollenbeck, C., Maj, P., Vitek, O., & Vitek, J. (2019, April 24). *On the impact of programming languages on code quality*. arXiv.org. [https://arxiv.org/abs/1901.10220](https://arxiv.org/abs/1901.10220)


    Vishal, M et al. (2020, March 30). *Is there a correlation between code comments and issues?: an exploratory study*. ACM Digital Library. [https://dl.acm.org/doi/10.1145/3341105.3374009](https://dl.acm.org/doi/10.1145/3341105.3374009)


    Aranda, J., & Venolia, G. (2009). *The secret life of bugs: Going past the errors and omissions in software repositories*. Research Gate. [https://www.researchgate.net/publication/221555113_The_secret_life_of_bugs_Going_past_the_errors_and_omissions_in_software_repositories](https://www.researchgate.net/publication/221555113_The_secret_life_of_bugs_Going_past_the_errors_and_omissions_in_software_repositories)
