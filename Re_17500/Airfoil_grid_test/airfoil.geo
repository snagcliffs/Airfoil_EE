// Description of airfoil
c = 1;
m = 0.04;
p = 0.4;
t = 0.12;
x_nose = -0.25;

// Description of domain
H = 2.5;// Domain height
h = 0.5; // c-curve radius
lf = 1; // Length of short field
Lf = 6; // Length of far field
Li = 2; // Length of inlet
wake_high = 2; // as fraction of h

shift = 0.1;

// Grid sizes along airfoil
n_lead = 15; // no. of points along airfoil containing leading edge
n_mid = 11;  // no. of points along center of airfoil 
n_tail = 33;  // no. of points along airfoil trailing edge 

n_norm = 33;  // no. of points normal to airfoil in inner radius
n_Norm = 21;  // no. of points normal to airfoil in outer radius
n_wake = 49;  //no. of points in the wake
n_Wake = 33;  //no. of points in the wake

//Progression ratios
p_tail = 1/1.05; // along the airfoil (1st part) containing trailing edge
p_lead = 1.2;  // along the airfoil (2nd part) containing leading edge
p_mid = 1.1;  // along the airfoil (2nd part) containing leading edge
p_norm = 1.1;  // normal to airfoil
p_Norm = 1.1;  // normal to airfoil
p_wake = 1.05; //  along the wake
p_Wake = 1.05; //  along the wake

// Local density (not used)
lc1 = 1.0;

Mesh.Smoothing = 0;

theta_tail = 80*Pi/180;
alpha = 5;

airfoil_lead_end = 100;
airfoil_mid_end = 350;

// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------

// Front and back of airfoil
Point(0) = {x_nose, 0., 0., lc1};
Point(1000) = {c+x_nose, 0., 0., lc1};

// Airofil surface
For i In {1:999}
    
    x = (i/1000)*c;
    yt = 5*t*(0.2969*Sqrt(x)-0.126*x-0.3516*x^2+0.2843*x^3-0.1036*x^4);

    If (x<p*c)
        yc = m/p^2*(2*p*(x/c)-(x/c)^2);
        dyc = 2*m/p^2*(p-x/c);
    EndIf

    If (x>=p*c)
        yc = m/(1-p)^2*(1-2*p+2*p*(x/c)-(x/c)^2);
        dyc = 2*m/(1-p)^2*(p-x/c);
    EndIf

    theta = Atan(dyc);

    xu = x - yt*Sin(theta) + x_nose;
    yu = yc + yt*Cos(theta);

    xl = x + yt*Sin(theta) + x_nose; 
    yl = yc - yt*Cos(theta);

    Point(i) = {xu, yu,  0.0 , lc1};
    Point(1000+i) = {xl , yl,  0.0 , lc1};

EndFor

allPoints[] = Point "*" ;
Rotate {{0, 0, 1}, {0, 0, 0}, -alpha*Pi/180} { Point{ allPoints[] } ; }

// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------

// Angle between (0,0) and front corner
theta_inlet = Atan(H/Li);

// Center of inlet curve
Point(2400) = {0,0,0,lc1};

// Ends of inner inlet c-curve
Point(2401) = {0,h,0,lc1};
Point(2402) = {-h*Cos(theta_inlet),h*Sin(theta_inlet),0,lc1};
Point(2403) = {-h,0,0,lc1};
Point(2404) = {-h*Cos(theta_inlet),-h*Sin(theta_inlet),0,lc1};
Point(2405) = {0,-h,0,lc1};

// Other points
Point(2406) = {-Li,H,0,lc1};
Point(2407) = {-Li,0,0,lc1};
Point(2408) = {-Li,-H,0,lc1};
Point(2409) = {-1,H,0,lc1};
Point(2410) = {-1,-H,0,lc1};

//Point(2411) = {x_nose+c+H/Tan(theta_tail),H,0,lc1};
Point(2411) = {x_nose+c+h/Tan(theta_tail),H,0,lc1};
Point(2412) = {x_nose+c+h/Tan(theta_tail),h,0,lc1};
Point(2413) = {x_nose+c+h/Tan(theta_tail),-h,0,lc1};
Point(2414) = {x_nose+c+h/Tan(theta_tail),-H,0,lc1};

Point(2415) = {x_nose+c+lf+H/Tan(theta_tail),H,0,lc1};
Point(2416) = {x_nose+c+lf+h/Tan(theta_tail),h+(wake_high-1)*lf/Lf*h,0,lc1};
Point(2417) = {x_nose+c+lf,-shift,0,lc1};
Point(2418) = {x_nose+c+lf+h/Tan(theta_tail),-h-(wake_high-1)*lf/Lf*h,0,lc1};
Point(2419) = {x_nose+c+lf+H/Tan(theta_tail),-H,0,lc1};

Point(2420) = {Lf+x_nose+c,H,0,lc1};
Point(2421) = {Lf+x_nose+c,wake_high*h-shift,0,lc1};
Point(2422) = {Lf+x_nose+c,-shift,0,lc1};
Point(2423) = {Lf+x_nose+c,-wake_high*h-shift,0,lc1};
Point(2424) = {Lf+x_nose+c,-H,0,lc1};

// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------

// Airfoil boundaries

Line(1) = {0:airfoil_lead_end}; Transfinite Curve {1} = n_lead Using Progression p_lead;
Line(2) = {0, 1001:1000+airfoil_lead_end}; Transfinite Curve {2} = n_lead Using Progression p_lead;
Line(3) = {airfoil_lead_end:airfoil_mid_end}; Transfinite Curve {3} = n_mid Using Progression p_mid;
Line(4) = {1000+airfoil_lead_end:1000+airfoil_mid_end}; Transfinite Curve {4} = n_mid Using Progression p_mid;
Line(5) = {airfoil_mid_end:1000}; Transfinite Curve {5} = n_tail Using Progression p_tail;
Line(6) = {1000+airfoil_mid_end:1999,1000}; Transfinite Curve {6} = n_tail Using Progression p_tail;

// C-curve
Circle(7) = {2403,2400,2402}; Transfinite Curve {7} = n_lead;
Circle(8) = {2403,2400,2404}; Transfinite Curve {8} = n_lead;
Circle(9) = {2402,2400,2401}; Transfinite Curve {9} = n_mid;
Circle(10) = {2404,2400,2405}; Transfinite Curve {10} = n_mid;

// Inlet
Line(11) = {2407,2406}; Transfinite Curve {11} = n_lead;
Line(12) = {2407,2408}; Transfinite Curve {12} = n_lead;
Line(13) = {2406,2409}; Transfinite Curve {13} = n_mid;
Line(14) = {2408,2410}; Transfinite Curve {14} = n_mid;

Line(15) = {2403,2407}; Transfinite Curve {15} = n_Norm Using Progression p_Norm;
Line(16) = {2402,2406}; Transfinite Curve {16} = n_Norm Using Progression p_Norm;
Line(17) = {2404,2408}; Transfinite Curve {17} = n_Norm Using Progression p_Norm;
Line(18) = {2401,2409}; Transfinite Curve {18} = n_Norm Using Progression p_Norm;
Line(19) = {2405,2410}; Transfinite Curve {19} = n_Norm Using Progression p_Norm;
Line(20) = {0,2403}; Transfinite Curve {20} = n_norm Using Progression p_norm;
Line(21) = {airfoil_lead_end,2402}; Transfinite Curve {21} = n_norm Using Progression p_norm;
Line(22) = {1000+airfoil_lead_end,2404}; Transfinite Curve {22} = n_norm Using Progression p_norm;
Line(23) = {airfoil_mid_end,2401}; Transfinite Curve {23} = n_norm Using Progression p_norm;
Line(24) = {1000+airfoil_mid_end,2405}; Transfinite Curve {24} = n_norm Using Progression p_norm;

Line(25) = {2409,2411}; Transfinite Curve {25} = n_tail;
Line(26) = {2401,2412}; Transfinite Curve {26} = n_tail;
Line(27) = {2405,2413}; Transfinite Curve {27} = n_tail;

Line(28) = {2410,2414}; Transfinite Curve {28} = n_tail;
Line(29) = {1000,2412}; Transfinite Curve {29} = n_norm Using Progression p_norm;
Line(30) = {1000,2413}; Transfinite Curve {30} = n_norm Using Progression p_norm;
Line(31) = {2412,2411}; Transfinite Curve {31} = n_Norm Using Progression p_Norm;

Line(32) = {2413,2414}; Transfinite Curve {32} = n_Norm Using Progression p_Norm;
Line(33) = {2411,2415}; Transfinite Curve {33} = n_wake;
Line(34) = {2412,2416}; Transfinite Curve {34} = n_wake;
Line(35) = {1000,2417}; Transfinite Curve {35} = n_wake Using Progression p_wake;
Line(36) = {2413,2418}; Transfinite Curve {36} = n_wake;
Line(37) = {2414,2419}; Transfinite Curve {37} = n_wake;


Line(38) = {2417,2416}; Transfinite Curve {38} = n_norm Using Progression p_Norm;
Line(39) = {2417,2418}; Transfinite Curve {39} = n_norm Using Progression p_Norm;
Line(40) = {2416,2415}; Transfinite Curve {40} = n_Norm Using Progression p_Norm;
Line(41) = {2418,2419}; Transfinite Curve {41} = n_Norm Using Progression p_Norm;
Line(42) = {2415,2420}; Transfinite Curve {42} = n_Wake Using Progression p_Wake;
Line(43) = {2416,2421}; Transfinite Curve {43} = n_Wake Using Progression p_Wake;

Line(44) = {2417,2422}; Transfinite Curve {44} = n_Wake Using Progression p_Wake;
Line(45) = {2418,2423}; Transfinite Curve {45} = n_Wake Using Progression p_Wake;
Line(46) = {2419,2424}; Transfinite Curve {46} = n_Wake Using Progression p_Wake;

Line(47) = {2422,2421}; Transfinite Curve {47} = n_norm;// Using Progression p_Norm;
Line(48) = {2422,2423}; Transfinite Curve {48} = n_norm;// Using Progression p_Norm;
Line(49) = {2421,2420}; Transfinite Curve {49} = n_Norm Using Progression p_Norm;
Line(50) = {2423,2424}; Transfinite Curve {50} = n_Norm Using Progression p_Norm;


// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------

// Surfaces

// Outer blocks
Line Loop(1) = {7,-21,-1,20};  Plane Surface(1) = {1};
Line Loop(2) = {-8,-20,2,22};  Plane Surface(2) = {2};
Line Loop(3) = {9,-23,-3,21};  Plane Surface(3) = {3};
Line Loop(4) = {-22,4,24,-10};  Plane Surface(4) = {4};
Line Loop(5) = {23,26,-29,-5};  Plane Surface(5) = {5};
Line Loop(6) = {6,30,-27,-24};  Plane Surface(6) = {6};

Line Loop(7) = {16,13,-18,-9};  Plane Surface(7) = {7};
Line Loop(8) = {11,-16,-7,15};  Plane Surface(8) = {8};
Line Loop(9) = {-15,8,17,-12};  Plane Surface(9) = {9};
Line Loop(10) = {-17,10,19,-14};  Plane Surface(10) = {10};
Line Loop(11) = {18,25,-31,-26};  Plane Surface(11) = {11};
Line Loop(12) = {-19,27,32,-28};  Plane Surface(12) = {12};

Line Loop(13) = {31,33,-40,-34};  Plane Surface(13) = {13};
Line Loop(14) = {29,34,-38,-35};  Plane Surface(14) = {14};

Line Loop(15) = {35,39,-36,-30};  Plane Surface(15) = {15};
Line Loop(16) = {36,41,-37,-32};  Plane Surface(16) = {16};
Line Loop(17) = {40,42,-49,-43};  Plane Surface(17) = {17};
Line Loop(18) = {43,-47,-44,38};  Plane Surface(18) = {18};

Line Loop(19) = {44,48,-45,-39};  Plane Surface(19) = {19};
Line Loop(20) = {45,50,-46,-41};  Plane Surface(20) = {20};


Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};
Transfinite Surface {6};
Transfinite Surface {7};
Transfinite Surface {8};
Transfinite Surface {9};
Transfinite Surface {10};
Transfinite Surface {11};
Transfinite Surface {12};
Transfinite Surface {13};
Transfinite Surface {14};
Transfinite Surface {15};
Transfinite Surface {16};
Transfinite Surface {17};
Transfinite Surface {18};
Transfinite Surface {19};
Transfinite Surface {20};

Recombine Surface {1};
Recombine Surface {2};
Recombine Surface {3};
Recombine Surface {4};
Recombine Surface {5};
Recombine Surface {6};
Recombine Surface {7};
Recombine Surface {8};
Recombine Surface {9};
Recombine Surface {10};
Recombine Surface {11};
Recombine Surface {12};
Recombine Surface {13};
Recombine Surface {14};
Recombine Surface {15};
Recombine Surface {16};
Recombine Surface {17};
Recombine Surface {18};
Recombine Surface {19};
Recombine Surface {20};

Physical Line("Inlet") = {11,12};
Physical Line("Outlet") = {47,48,49,50};
Physical Line("Symm") = {13,14,25,28,33,37,42,46};
Physical Line("Wall") = {1,2,3,4,5,6};
Physical Surface("Fluid") = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

