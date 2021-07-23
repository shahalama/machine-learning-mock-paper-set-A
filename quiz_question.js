

const quiz = [
	{
		q:'FIND-S Algorithm starts from the most specific hypothesis and generalize it by considering only',
		options:['Negative','Positive','Negative and positive','None of the above'],
		answer:1
	},
	{
		q:'Which of the following does not include different learning methods',
		options:['Analogy','Introduction','Memorization','Deduction'],
		answer:1
	},
	{
		q:'Father of Machine Learning (ML)',
		options:['Geoffrey Chaucer','Geoffrey Hill','Geoffrey Everest Hinton','None of the above'],
		answer:2
	},
	{
		q:' FIND-S algorithm ignores',
		options:['Negative','Positive','both','None of the above'],
		answer:0
	},
	{
		q:'The Candidate-Elimination Algorithm represents the',
		options:['Solution Space','Version Space','Elimination Space','All of the Above'],
		answer:1
	},
	{
		q:'Inductive learning is based on the knowledge that if something happens a lot it is likely to be generally',
		options:['True','False'],
		answer:0
	},
	{
		q:'Which of the following is a good test dataset characteristic?',
		options:['Large enough to yield meaningful results','Is representative of the dataset as a whole','Both A and B','None of the above'],
		answer:2
	},
	{
		q:' Which of the following is not numerical functions in the various function representation of Machine Learning?',
		options:['Neural Network','Support Vector Machines','Case-based','Linear Regression'],
		answer:2
	},
	{
		q:'A drawback of the FIND-S is that it assumes the consistency within the training set ',
		options:['True','False'],
		answer:0
	},
	{
		q:'How machines learn? I. Training II. Validation III. Application ',
		options:[' I only','I & II Only','I,II, & III','II & III'],
		answer:2
	},
	{
		q:'What strategies can help reduce overfitting in decision trees? I Enforce a maximum depth for the tree II Enforce a minimum number of samples in leaf nodes III Pruning IV Make sure each leaf node is one pure class',
		options:[' All','(i), (ii) and (iii)','(i), (iii), (iv)','None'],
		answer:1
	},
	{
		q:'Which of the following is a widely used and effective machine learning algorithm based on the idea of bagging?',
		options:[' Decision Tree','Random Forest','Regression','Classification'],
		answer:1
	},
	{
		q:'To find the minimum or the maximum of a function, we set the gradient to zero because which of the following',
		options:[' Depends on the type of problem','The value of the gradient at extrema of a function is always zero','Both (A) and (B)','None of these'],
		answer:1
	},
	{
		q:'What is perceptron?',
		options:['A single layer feed-forward neural network with pre-processing','A neural network that contains feedback','A double layer auto-associative neural network','An auto-associative neural network'],
		answer:0
	},
	{
		q:'What is Neuro software?',
		options:['It is software used by Neurosurgeon','Designed to aid experts in real world','It is powerful and easy neural network','A software used to analyze neurons'],
		answer:2
	},
	{
		q:'Which is true for neural networks?',
		options:['Each node computes it‟s weighted input',' Node could be in excited state or non-excited state','It has set of nodes and connections','All of the above'],
		answer:3
	},
	{
		q:'Which of the following is true? Single layer associative neural networks do not have the ability to:- I Perform pattern recognition II Find the parity of a picture III Determine whether two or more shapes in a picture are connected or not ',
		options:[' (ii) and (iii)',' Only (ii)','All','None'],
		answer:0
	},
	{
		q:'The backpropagation law is also known as generalized delta rule ',
		options:[' True','False'],
		answer:0
	},
	{
		q:'A 3-input neuron has weights 1, 4 and 3. The transfer function is linear with the constant of proportionality being equal to 3. The inputs are 4, 8 and 5 respectively. What will be the output?',
		options:[' 139','153','162','160'],
		answer:1
	},
	{
		q:'What is back propagation?',
		options:['It is another name given to the curvy function in the perceptron','It is the transmission of error back through the network to allow weights to be adjusted so that the network can learn',' It is another name given to the curvy function in the perceptron','None of the above'],
		answer:1
	},
	{
		q:'Neural Networks are complex ___functions with many parameter ',
		options:['Linear','Non linear','Discreate','Exponential'],
		answer:0
	},
	{
		q:'In backpropagation rule, how to stop the learning process? ',
		options:[' No heuristic criteria exist','On basis of average gradient value','There is convergence involved','There is convergence involved'],
		answer:1
	},
	{
		q:'Which of the following is the consequence between a node and its predecessors while creating bayesian network?  ',
		options:[' Conditionally independent','Functionally dependent',' Both Conditionally dependant & Dependant','Dependent'],
		answer:0
	},
	{
		q:'Bayes rule can be used for:-',
		options:['Solving queries','Increasing complexity','Answering probabilistic query','Decreasing complexity'],
		answer:2
	},
	{
		q:'The bayesian network can be used to answer any query by using:-',
		options:[' Full distribution','Joint distribution',' Partial distribution','All of the above'],
		answer:1
	},
	{
		q:'Bayesian networks allow compact specification of:-',
		options:['Joint probability distributions','Belief','Propositional logic statements','All of the above'],
		answer:0
	},
	{
		q:'Which of the following is correct about the Naive Bayes? ',
		options:['  Assumes that all the features in a dataset are independent','Assumes that all the features in a dataset are equally important','Both','All of the above'],
		answer:2
	},
	{
		q:'Naïve Bayes Algorithm is a learning algorithm',
		options:['Supervised','Reinforcement','Unsupervised','None of these'],
		answer:0
	},
	{
		q:'Examples of Naïve Bayes Algorithm is/are ',
		options:[' Spam filtration','Sentimental analysis','Classifying articles',' All of the above'],
		answer:3
	},
	{
		q:'Disadvantages of Naïve Bayes Classifier: ',
		options:[' Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features',' It performs well in Multi-class predictions as compared to the other classifiers','Naïve Bayes is one of the fast and easy ML algorithms to predict a class of input','It is the most popular choice for text classification problems.'],
		answer:0
	},
	{
		q:'What are the area CLT comprised of? ',
		options:['Sample Complexity',' Computational Complexity','Mistake Bound','All of these'],
		answer:3
	},
	{
		q:'What area of CLT tells “How many examples we need to find a good hypothesis ?',
		options:['Sample Complexity',' Computational Complexity','Mistake Bound','None of these'],
		answer:0
	},
	{
		q:'What area of CLT tells “How much computational power we need to find a good hypothesis ? ',
		options:['Sample Complexity',' Computational Complexity','Mistake Bound','None of these'],
		answer:1
	},
	{
		q:'What area of CLT tells “How many mistakes we will make before finding a good hypothesis ?',
		options:['Sample Complexity',' Computational Complexity','Mistake Bound','None of these'],
		answer:2
	},
	{
		q:'The VC dimension of hypothesis space H1 is larger than the VC dimension of hypothesis space H2. Which of the following can be inferred from this? ',
		options:[' The number of examples required for learning a hypothesis in H1 is larger than the number of examples required for H2','The number of examples required for learning a hypothesis in H1 is smaller than the number of examples required for H2',' No relation to number of samples required for PAC learning.','None of above'],
		answer:0
	},
	{
		q:'For a particular learning task, if the requirement of error parameter changes from 0.1 to 0.01. How many more samples will be required for PAC learning?',
		options:['  Same','2 times','1000 times','10 times'],
		answer:3
	},
	{
		q:'Computational complexity of classes of learning problems depends on which of the following? ',
		options:['The size or complexity of the hypothesis space considered by learner','The accuracy to which the target concept must be approximated','The probability that the learner will output a successful hypothesis','All of these'],
		answer:3
	},
	{
		q:'What are the advantages of Nearest neighbour algo? ',
		options:['Training is very fast','Can learn complex target functions','Don‟t lose information',' All of these'],
		answer:3
	},
	{
		q:'What are the difficulties with k-nearest neighbor algo?',
		options:['Calculate the distance of the test case from all training cases',' Curse of dimensionality','Both A & B','None of these'],
		answer:2
	},
	{
		q:' What is/are true about Distance-weighted KNN? ',
		options:[' The weight of the neighbour is considered','The distance of the neighbour is considered',' Both A & B',' None of these'],
		answer:2
	},
	{
		q:'Genetic algorithm is a ',
		options:['Search technique used in computing to find true or approximate solution to optimization and search problem','Sorting technique used in computing to find true or approximate solution to optimization and sort problem','Both A & B','None of these'],
		answer:0
	},
	{
		q:'When would the genetic algorithm terminate?',
		options:['Maximum number of generations has been produced','Satisfactory fitness level has been reached for the solution','Both A & B','None of these'],
		answer:2
	},
	{
		q:'The algorithm operates by iteratively updating a pool of hypotheses, called the  ',
		options:['Population',' Fitness','None of these'],
		answer:0
	},
	{
		q:'What is the correct representation of GA?',
		options:[' GA(Fitness, Fitness_threshold, p)','GA(Fitness, Fitness_threshold, p, r )',' GA(Fitness, Fitness_threshold, p, r, m)',' GA(Fitness, Fitness_threshold)'],
		answer:2
	},
	{
		q:'Genetic operators include ',
		options:[' Crossover','Mutation','Both A & B','None of these'],
		answer:2
	},
	{
		q:'ILP stand for ',
		options:['Inductive Logical programming','Inductive Logic Programming',' Inductive Logical Program',' Inductive Logic Program'],
		answer:1
	},
	{
		q:' Ground literal is a literal that ',
		options:['Contains only variables','does not contains any functions','does not contains any variables',' Contains only functions'],
		answer:2
	},
	{
		q:'Features of Reinforcement learning  ',
		options:[' Set of problem rather than set of techniques','RL is training by reward and','RL is learning from trial and error with the','All of these'],
		answer:3
	},
	{
		q:' What is/are the problem solving methods for RL? ',
		options:['Dynamic programming','Monte Carlo Methods','Temporal-difference learning',' All of these'],
		answer:3
	},
	{
		q:' Which among the following is not a necessary feature of a reinforcement learning solution to a learning problem? ',
		options:['  exploration versus exploitation dilemma','trial and error approach to learning','learning based on rewards','representation of the problem as a Markov Decision Process'],
		answer:3
	}

]