// Geometría pilote
// author: Juan Gomez
H  = 20.0;   // Distancia suelo-fondo del modelo
W  = 20.0;   // Semiancho del modelo.
E  = 15.0;   // Semiancho del deposito.
L  =  10.0;   // Longitud del pilote.
e1 =  2.0;
e2 =  3.0;
e3 =  3.0;
e4 =  4.0;
e5 = 10.0;
r  = 0.45;  // Radio del pilote.
Z  = 2.0;   // Semiancho de la zapata.
f  = e1+e2+e3+e4+e5-L;  // Distancia fondo pilote-fondo deposito
t = 0.50;               // Espesor de la zapata.
hc = 0.50;              // Semiancho de la zona de carga.
d  = 0.50;    // Semi-distancia entre caras de pilotes
g  = 0.25;    // Distancia del pilote al extremo de la zapata.
c  = W/10; 						// Semilla para los elementos
//
// Define vertex points
// 						
Point(1)   = {-W   , 0.0 ,   0, c};		        // {x,y,z, size}
Point(2)   = { W   , 0.0 ,   0, c};
Point(3)   = { W   , H+e1+e2+e3+e4+e5  , 0, c};
Point(4)   = { E   , H+e1+e2+e3+e4+e5   , 0, c/5};
Point(5)   = { E   , H   , 0, c/5};
Point(6)   = {-E   , H   , 0, c/5};
Point(7)   = {-E   , H+e1+e2+e3+e4+e5   , 0, c/5};
Point(8)   = {-W   , H+e1+e2+e3+e4+e5  , 0, c};
Point(9)   = { E   , H+e2+e3+e4+e5   , 0, c/5};
Point(10)  = { E   , H+e3+e4+e5   , 0, c/5};
Point(11)  = { E   , H+e4+e5   , 0, c/5};
Point(12)  = { E   , H+e5   , 0, c/5};
Point(13)  = {-E   , H+e2+e3+e4+e5   , 0, c/5};
Point(14)  = {-E   , H+e3+e4+e5   , 0, c/5};
Point(15)  = {-E   , H+e4+e5   , 0, c/5};
Point(16)  = {-E   , H+e5   , 0, c/5};
Point(17)  = { Z   , H+e1+e2+e3+e4+e5   , 0, c};
Point(18)  = {-Z   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(19)  = { Z   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(20)  = {-Z   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(21)  = { hc   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(22)  = {-hc   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(23)  = { 0.0  , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(24)  = {  (d+2*r)   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(25)  = {  (d+2*r)   , H+e2+e3+e4+e5   , 0, c/10};Point(26)  = {  (d+2*r)   , H+e3+e4+e5   , 0, c/10};Point(27)  = {  (d+2*r)   , H+e4+e5   , 0, c/10};Point(28)  = {  (d+2*r)   , H+f  , 0, c/10};
Point(29)  = {  d   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(30)  = {  d   , H+e2+e3+e4+e5   , 0, c/10};Point(31)  = {  d   , H+e3+e4+e5   , 0, c/10};Point(32)  = {  d   , H+e4+e5   , 0, c/10};Point(33)  = {  d   , H+f  , 0, c/10};
Point(34)  = { -d   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(35)  = {  -d   , H+e2+e3+e4+e5   , 0, c/10};Point(36)  = {  -d   , H+e3+e4+e5   , 0, c/10};Point(37)  = {  -d   , H+e4+e5   , 0, c/10};Point(38)  = {  -d   , H+f  , 0, c/10};
Point(39)  = {  -(d+2*r)   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(40)  = {  -(d+2*r)   , H+e2+e3+e4+e5   , 0, c/10};Point(41)  = {  -(d+2*r)   , H+e3+e4+e5   , 0, c/10};Point(42)  = {  -(d+2*r)   , H+e4+e5    , 0, c/10};Point(43)  = {  -(d+2*r)   , H+f  , 0, c/10};
//
// Lineas
//
Line(1) = {1, 2};					// {Initial_point, end_point}
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 9};
Line(5) = {9, 10};
Line(6) = {10, 11};
Line(7) = {11, 12};
Line(8)  = {12,  5};
Line(9)  = {5, 6};
Line(10) = {6, 16};
Line(11) = {16, 15};
Line(12) = {15, 14};
Line(13) = {14, 13};
Line(14) = {13, 7};
Line(15) = {7, 8};
Line(16) = {8, 1};
Line(17) = {4, 17};
Line(18) = {18, 7};
Line(19) = {17, 19};
Line(20) = {18, 20};
Line(21) = {19, 21};
Line(22) = {20, 22};
Line(23) = {21, 23};
Line(24) = {22, 23};
Line(25) = {17, 24};
Line(26) = {18, 39};
Line(27) = {24, 29};
Line(28) = {39, 34};
Line(29) = {34, 29};
Line(30) = {24, 25};
Line(31) = {25, 26};
Line(32) = {26, 27};
Line(33) = {27, 28};
Line(34) = {29, 30};
Line(35) = {30, 31};
Line(36) = {31, 32};
Line(37) = {32, 33};
Line(38) = {33, 28};
Line(39) = {34, 35};
Line(40) = {35, 36};
Line(41) = {36, 37};
Line(42) = {37, 38};
Line(43) = {39, 40};
Line(44) = {40, 41};
Line(45) = {41, 42};
Line(46) = {42, 43};
Line(47) = {43, 38};
Line(48) = {13, 40};
Line(49) = {14, 41};
Line(50) = {15, 42};
Line(51) = {35, 30};
Line(52) = {36, 31};
Line(53) = {37, 32};
Line(54) = {25, 9};
Line(55) = {26, 10};
Line(56) = {27, 11};
Line(57) = {16, 12};
//
// Loops
//
Line Loop(1)  = {1 , 2 , 3 , 4, 5, 6, 7, 8, 9, 10, 11 , 12, 13, 14 , 15 , 16};     // Externo
Line Loop(2)  = {-9, -8, -57, -10};     // Estrato inferior completo
Line Loop(3)  = {57, -7, -56, 33, -38, -37, -53, 42, -47, -46, -50, -11};     // Estrato 4 completo
Line Loop(4)  = {56, -6, -55, 32};     // Estrato 3 derecho
Line Loop(5)  = {55, -5, -54, 31};     // Estrato 2 derecho
Line Loop(6)  = {54, -4, 17, 25, 30};     // Estrato 1 derecho
Line Loop(7)  = {50, -45, -49, -12};     // Estrato 3 izquierdo
Line Loop(8)  = {49, -44, -48, -13};     // Estrato 2 izquierdo
Line Loop(9)  = {48, -43, -26, 18, -14};     // Estrato 1 izquierdo
Line Loop(10)  = {53, -36, -52, 41};       // Estrato 3 centro
Line Loop(11)  = {52, -35, -51, 40};     // Estrato 2 centro
Line Loop(12)  = {51, -34, -29, 39};     // Estrato 1 centro
Line Loop(13)  = {47, -42, -41, -40, -39, -28, 43, 44, 45, 46};     // Pilote izquierdo
Line Loop(14)  = {38, -33, -32, -31, -30, 27, 34, 35, 36, 37};     // Pilote derecho
Line Loop(15)  = {19, 21, 23, -24, -22, -20, 26, 28, 29, -27, -25};     // Zapata//
// planos
// 				
Plane Surface(10)   = {1};
Plane Surface(20)   = {2};
Plane Surface(30)   = {3};
Plane Surface(40)   = {4};
Plane Surface(50)   = {5};
Plane Surface(60)   = {6};
Plane Surface(70)   = {7};
Plane Surface(80)   = {8};
Plane Surface(90)   = {9};
Plane Surface(110)   = {10};
Plane Surface(111)   = {11};
Plane Surface(112)   = {12};
Plane Surface(113)   = {13};
Plane Surface(114)   = {14};
Plane Surface(115)   = {15};//
// Superficies físicas
//
Physical Surface(100) = {10};           // Semi-espacio
Physical Surface(200) = {20};          // Estrato 5 completo
Physical Surface(300) = {30};          // Estrato 4 completo
Physical Surface(400) = {40, 70, 110}; // Estrato 3 completo
Physical Surface(500) = {50, 80, 111}; // Estrato 2 completo
Physical Surface(600) = {60, 90, 112}; // Estrato 1 completo
Physical Surface(700) = {113, 114, 115}; // Zapata + Pilotes
//
// Lineas físicas
//
Physical Line(1000) = {1};
Physical Line(2000) = {2 , 16};
Physical Line(3000) = {23};
Physical Line(4000) = {24};
