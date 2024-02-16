// Geometría pilote
// author: Juan Gomez
H  = 20.0;   // Distancia suelo-fondo del modelo
W  = 20.0;   // Semiancho del modelo.
E  = 15.0;   // Semiancho del deposito.
L  =  10.0;  // Longitud del pilote.
e1 =  2.0;
e2 =  3.0;
e3 =  3.0;
e4 =  4.0;
e5 = 10.0;
r  = 0.25;  // Radio del pilote.
Z  = 2.0;   // Semiancho de la zapata.
f  = e1+e2+e3+e4+e5-L;  // Distancia fondo pilote-fondo deposito
t = 0.50;               // Espesor de la zapata.
hc = 0.50;              // Semiancho de la zona de carga.
c  = W/10; 						// Semilla para los elementos
//
// Define vertex points
// 						
Point(1)   = {-W   , 0.0 ,   0, c};		        // {x,y,z, size}
Point(2)   = { W   , 0.0 ,   0, c};
Point(3)   = { W   , H+e1+e2+e3+e4+e5  , 0, c};
Point(4)   = { E   , H+e1+e2+e3+e4+e5   , 0, c};
Point(5)   = { E   , H   , 0, c};
Point(6)   = {-E   , H   , 0, c};
Point(7)   = {-E   , H+e1+e2+e3+e4+e5   , 0, c};
Point(8)   = {-W   , H+e1+e2+e3+e4+e5  , 0, c};
Point(9)   = { E   , H+e2+e3+e4+e5   , 0, c};
Point(10)  = { E   , H+e3+e4+e5   , 0, c};
Point(11)  = { E   , H+e4+e5   , 0, c};
Point(12)  = { E   , H+e5   , 0, c};
Point(13)  = {-E   , H+e2+e3+e4+e5   , 0, c};
Point(14)  = {-E   , H+e3+e4+e5   , 0, c};
Point(15)  = {-E   , H+e4+e5   , 0, c};
Point(16)  = {-E   , H+e5   , 0, c};
Point(17)  = { Z   , H+e1+e2+e3+e4+e5   , 0, c};
Point(18)  = {-Z   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(19)  = { Z   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(20)  = {-Z   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(21)  = { hc   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(22)  = {-hc   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(23)  = { 0.0  , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(24)  = { r   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(25)  = { r   , H+e2+e3+e4+e5   , 0, c/10};
Point(26)  = { r   , H+e3+e4+e5   , 0, c/10};
Point(27)  = { r   , H+e4+e5   , 0, c/10};
Point(28)  = { r   , H+f   , 0, c/10};
Point(29)  = {-r   , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(30)  = {-r   , H+e2+e3+e4+e5   , 0, c/10};
Point(31)  = {-r   , H+e3+e4+e5   , 0, c/10};
Point(32)  = {-r   , H+e4+e5   , 0, c/10};
Point(33)  = {-r   , H+f   , 0, c/10};
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
Line(26) = {18, 29};
Line(27) = {29, 24};
Line(28) = {24, 25};
Line(29) = {25, 26};
Line(30) = {26, 27};
Line(31) = {27, 28};
Line(32) = {29, 30};
Line(33) = {30, 31};
Line(34) = {31, 32};
Line(35) = {32, 33};
Line(36) = {33, 28};
Line(37) = {13, 30};
Line(38) = {14, 31};
Line(39) = {15, 32};
Line(40) = {25,  9};
Line(41) = {26, 10};
Line(42) = {27, 11};
Line(43) = {16, 12};
//
// Loops
//
Line Loop(1)  = {1 , 2 , 3 , 4, 5, 6, 7, 8, 9, 10, 11 , 12, 13, 14 , 15 , 16};     // Externo
Line Loop(2)  = {-9 , -8 , -43, -10};                                             // Estrato 5 inferior completo
Line Loop(3)  = {43, -7, -42, 31, -36, -35, -39, -11};                            // Estrato 4 inferior
Line Loop(4)  = {42, -6, -41, 30};                                                // Estrato 3 derecho
Line Loop(5)  = {41, -5, -40, 29};                                                // Estrato 2 derecho
Line Loop(6)  = {40, -4, 17, 25, 28};                                              // Estrato 1 derecho
Line Loop(7)  = {39, -34, -38, -12};                                              // Estrato 3 izquierdo
Line Loop(8)  = {38, -33, -37, -13};                                              // Estrato 2 izquierdo
Line Loop(9)  = {37, -32, -26, 18, -14};                                          // Estrato 1 izquierdo
Line Loop(10)  = {36, -31, -30, -29, -28, -27, 32, 33, 34, 35};                    // Pilote
Line Loop(11)  = {19, 21, 23, -24, -22, -20, 26, 27, -25};                        // Zapata
//
// planos 					// {Id_Loop}
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
Plane Surface(101)  = {10};
Plane Surface(102)  = {11};//
// Superficies físicas
//
Physical Surface(100) = {10};   // Semi-espacio
Physical Surface(200) = {20};   // Estrato 5 completo
Physical Surface(300) = {30};   // Estrato 4 completo
Physical Surface(400) = {40};   // Estrato 3 derecho
Physical Surface(500) = {50};   // Estrato 2 derecho
Physical Surface(600) = {60};   // Estrato 1 derecho
Physical Surface(700) = {70};   // Estrato 3 izquierdo
Physical Surface(800) = {80};   // Estrato 2 izquierdo
Physical Surface(900) = {90};   // Estrato 1 izquierdo
Physical Surface(111) = {101};   // Pilote 
Physical Surface(122) = {102};  // Zapata
//
// Lineas físicas
//
Physical Line(1000) = {1};
Physical Line(2000) = {2 , 16};
Physical Line(3000) = {23};
Physical Line(4000) = {24};