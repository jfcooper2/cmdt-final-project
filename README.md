## CS 714 Project - Erosion

#### Repo Structure

Most of the code you'll be looking for is in `erosion` and 
`erosion_deposition`. The former is the model with all deposition
terms removed to see the behavior with only an erosive term. The 
latter is where most of the project is. The `main.py` script 
contains all of the code ran for these simulations as described 
in the assocaited paper. All results for each are stored in `img` 
directories in each subfolder. Futher executions of the code will 
create a new directory titled `itr` followed by a number. The 
same directory in the base of this repository is from earlier 
iterations of the code. Check those out if you want to see really 
wonky bugs!

#### What does the code give me?

In each `itr` directory, there will be a sequence of images from 
the simulation created. A parameters json file will also be created 
with all parameters from a give run except for the source terms both 
at the boundary and in the interior. Lastly, and most importantly, 
there will be an animation created compiling a large number of 
iterations into a nice grey-scale timelapse showing the neat powers
of erosion.

#### What's in oldcode?

Anything that was written that gave buggy or slow results.
