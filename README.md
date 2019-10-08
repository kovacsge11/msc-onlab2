# msc-onlab2
My code to subject "Önálló laboratórium 2" at masters.
I used the framework implemented by Peter Salvi: https://github.com/salvipeter/sample-framework/tree/757a0b1af2f4a17fb536777664c68145f386b65e.

## Rule checking
According to the publication [T-Splines and T-NURCCs by Sederberg et al. 2003](https://www.researchgate.net/publication/234827617_T-splines_and_T-NURCCs) two rules of T-mesh have to be checked in order to validate the correctness of the topology. 

**Rule 1:** The sum of knot intervals on opposing edges of any face must be equal.

**Rule 2:** If a T-junction on one edge of a face can be connected to a T-junction on an opposing edge of the face (thereby splitting the face into two faces) without violating Rule 1, that edge must be included in the T-mesh.

I based my implementation on [T-spline simplification and local refinement by Sederberg et al. 2004](https://www.researchgate.net/publication/234780696_T-spline_simplification_and_local_refinement) partly, too. I implemented the control net drawing in such a way, that the T-mesh topology created this way is never going to fail the rule written in this publication:

>Knot vectors si (4) and ti (5) for the blending function of Pi are determined as follows. (si2; ti2) are the knot coordinates of Pi. Consider a ray in parameter space R(a) = (si2 +a; ti2). Then si3 and si4 are the s coordinates of the first two s-edges intersected by the ray (not including the initial (si2; ti2)). By s-edge, we mean a vertical line segment of constant s. The other knots in si and ti are found in like manner.

I bind two consecutive points in a row or column if the fourth element of the knot vector of the first point in order is the knot value of the point itself = the third element of its knot vector (for rows and columns accordingly). This way **Rule 2** is always going to apply to my implementation.
