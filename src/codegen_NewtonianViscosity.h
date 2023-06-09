  double t2 = nu*2.0;
  double t3 = f_1_2+f_2_1;
  double t4 = nu*t3;
  double t5 = f_1_3+f_3_1;
  double t6 = nu*t5;
  double t7 = f_2_3+f_3_2;
  double t8 = nu*t7;
  H(0,0) = t2;
  H(1,1) = nu;
  H(1,3) = nu;
  H(2,2) = nu;
  H(2,6) = nu;
  H(3,1) = nu;
  H(3,3) = nu;
  H(4,4) = t2;
  H(5,5) = nu;
  H(5,7) = nu;
  H(6,2) = nu;
  H(6,6) = nu;
  H(7,5) = nu;
  H(7,7) = nu;
  H(8,8) = t2;
  stress(0) = f_1_1*nu*2.0;
  stress(1) = t4;
  stress(2) = t6;
  stress(3) = t4;
  stress(4) = f_2_2*nu*2.0;
  stress(5) = t8;
  stress(6) = t6;
  stress(7) = t8;
  stress(8) = f_3_3*nu*2.0;
