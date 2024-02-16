// Input .geo for dam
// author: Juan Gomez
H  = 5.0;
W  = 5.0;
c  = W/10; 						// for size elements

// Define vertex points 						
Point(1) = {0.0   , 0.0 , 0, c};		        // {x,y,z, size}
Point(2) = {W     , 0.0 , 0, c};
Point(3) = {W     , H/2 , 0, c};
Point(4) = {W     , H   , 0, c};
Point(5) = {0.0   , H   , 0, c};
Point(6) = {0.0   , H/2 , 0, c};

// Define boundary lines
Line(1) = {1, 2};					// {Initial_point, end_point}
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {3, 6};
//
// Joint Lines
//
Line Loop(8)  = {1 , 2 , 7 , 6};
Line Loop(10) = {3 , 4 , 5 ,-7};
//
// surface for mesh 					// {Id_Loop}
//
Plane Surface(9)  = {8};
Plane Surface(11) = {10};
//
//
// Physical surface. Two material
//
Physical Surface(1000) = {9};
Physical Surface(2000) = {11};
//
//Physical line. Boundary
//
Physical Line(100) = {1};
Physical Line(200) = {2 , 3, 5 ,6};
Physical Line(300) = {4};