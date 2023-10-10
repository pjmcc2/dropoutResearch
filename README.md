# Dropout with Online Difficulty-Scaling 

## An in-progress research project


What if the probabilities in a dropout layer were altered with respect to the network's performance?
This project aims to answer that question, if only a little. This version of the popular dropout algorithm is nicknamed "Tsetlin"
because I was inpired by the reinforcement learning-based Tsetlin Machines (see Granmo and similar) to use "voting" to affect the
difficulty faced by the network in the form of altering the dropout probabilities. More dropped-out values result in a harder classification
problem for the network, and visa-versa for fewer. Essentially, my idea was as follows: if dropout seeks to combat overfitting, and networks increase in accuracy (or some other metric)
as they train, what if dropping out increased as they trained? This idea was certainly not original, so I reviewed exisiting literature for inspiration and to avoid plagiarism.
In that process I read Morerio et al's paper on Curriculum (or scheduled) dropout. This uses the same philosophy of increasing in difficulty throughout the training lifespan. 
Their expirements yielded good results, so I was hopeful my version would too! 

