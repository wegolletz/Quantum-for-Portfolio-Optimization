# Project 2: Quantum for Portfolio Optimization

**Team name**: PortfoQ

**Members**:
- Czcibor Ciostek, WISER Enrollment ID: gst-gzfvWcg4u1SdUnN
- Weronika Golletz, WISER Enrollment ID: gst-8Mu81qj8JuZZzcZ

**Project Summary**:

1. We began by reviewing the mathematical formulation provided, focusing on binary decision vari-
ables, linear constraints, and the quadratic objective. We analyzed and understood the provided
codebase and results from the GitHub repository. Rather than starting from scratch, we aimed
to build upon this foundation to conduct further research and experimentatio

2. We identified key opportunities for improving the Variational Quantum Eigensolver (VQE)
optimization process, with a particular focus on:
     - The choice and structure of the ansatz: For the 31-bond problem, we
    selected the TwoLocal and bfcd ansatzes with CVaR parameter α = 0.1, repetition depth
    r = 2, and bilinear entanglement. This configuration achieved the most optimal solutions
    and fastest convergence among the tested parameter values, and was used in subsequent
    analyses.
    - The penalty parameter in constraint embedding:  We tested $\lambda \in \{1.1,\, 1.5,\, 2.0,\, 2.5\}$. The lowest penalty $\lambda=1.1$ yielded the best convergence and optimization results for both ansatzes. However, constraint satisfaction was 80\% for TwoLocal at $\lambda=1.1$, while $\lambda=1.5$ gave the highest satisfaction rate for bfcd (90\%).
    - The circuit initialization strategy:  We compared the default initialization of all parameters to $\pi/3$ against random initialization. The $\pi/3$ initialization proved to be a reasonable choice: it gave the highest constraint satisfaction rates (90\% for TwoLocal, 70\% for bfcd), while random initialization produced mixed results—sometimes improving convergence, but often not.

3. To test scalability and performance, we developed a custom generator for producing problem instances of the same type as the original (binary quadratic problems with linear inequality constraints), but for varying numbers of bonds. This allowed us to
   - Study scaling behavior across problem sizes -- we analyzed problems with $15,\,20,\,25,\,30$ bonds, and found that while solution time increases predictably with size, the relative gap to the optimum also grows, with variance widening for larger instances.
   - Perform statistical evaluations across multiple runs and configurations -- comparing 100 different 30-bond problems to 100 attempts on the same problem shows that certain problem instances are substantially harder, with some exhibiting objective values more than 50\% worse than the optimum.
4. Motivated by recent literature, we also implemented a new ansatz -- the Heat Exchange ansatz.
5. Finally, we outlined future research directions to further improve solution quality and scalability.



**All project details are described in report.pdf**
