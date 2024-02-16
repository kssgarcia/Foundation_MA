// Geometría pilote
// Autor: Juan Gomez
H  = 10.0;   // Distancia deposito-fondo del modelo
W  = 30.0;   // Semiancho del modelo.
E  = 15.0;   // Semiancho del deposito.
e1 =  2.0;
e2 =  3.0;
e3 =  3.0;
e4 =  4.0;
e5 = 10.0;
Z  =  2.0;    // Semiancho de la zapata.
sc  = 0.50;   // Semiancho de la zona de carga.(huella de la columna sobre la zapata)
t   = 0.50;  // Espesor de la zapata
c  = W/10;    // Semilla para los elementos
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
Point(18)  = {0.0  , H+e1+e2+e3+e4+e5   , 0, c/10};
Point(19)  = {-Z   , H+e1+e2+e3+e4+e5   , 0, c};
Point(20)  = { Z   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(21)  = {-Z   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(22)  = { sc   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(23)  = {-sc   , H+e1+e2+e3+e4+e5+t   , 0, c};
Point(24)  = {0.0  , H+e1+e2+e3+e4+e5+t   , 0, c/10};
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
Line(18) = {7, 19};
Line(19) = {17, 18};
Line(20) = {19, 18};
Line(21) = {13, 9};
Line(22) = {14, 10};
Line(23) = {15, 11};
Line(24) = {16, 12};
Line(25) = {17, 20};
Line(26) = {19, 21};
Line(27) = {20, 22};
Line(28) = {21, 23};
Line(29) = {23, 24};
Line(30) = {22, 24};
//
// Loops
//
Line Loop(1)  = {1 , 2 , 3 , 4, 5, 6, 7, 8, 9, 10, 11 , 12, 13, 14 , 15 , 16};     // Externo   (Roca)
Line Loop(2)  = {-9 , -8 , -24, -10};                                              // Estrato 5 (Inferior)
Line Loop(3)  = {24, -7, -23, -11};                                                // Estrato 4
Line Loop(4)  = {23, -6, -22, -12};                                                // Estrato 3
Line Loop(5)  = {22, -5, -21, -13};                                                // Estrato 2
Line Loop(6)  = {21, -4, 17, 19, -20, -18, -14};                                   // Estrato 1 (Superficial)
Line Loop(7)  = {20, -19, 25, 27, 30, -29, -28, -26};                              // Zapata
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
//
// Superficies físicas
//
Physical Surface(100) = {10}; // Semi-espacio
Physical Surface(200) = {20}; // Estrato 5 completo
Physical Surface(300) = {30}; // Estrato 4 completo
Physical Surface(400) = {40}; // Estrato 3 completo
Physical Surface(500) = {50}; // Estrato 2 completo
Physical Surface(600) = {60}; // Estrato 1 completo
Physical Surface(700) = {70}; // Zapata
//
// Lineas físicas
//
Physical Line(1000) = {1};
Physical Line(2000) = {2 , 16};
Physical Line(3000) = {29};
Physical Line(4000) = {30};