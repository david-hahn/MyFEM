// from MuPad material_linElast_derivs.mn
  double t1 = f_1_1+f_2_2+f_3_3-3.0;
  double t2 = f_1_2+f_2_1;
  double t3 = f_1_3+f_3_1;
  double t4 = f_2_3+f_3_2;
  stress_la[0] = t1;
  stress_la[4] = t1;
  stress_la[8] = t1;
  stress_mu[0] = f_1_1*2.0-2.0;
  stress_mu[1] = t2;
  stress_mu[2] = t3;
  stress_mu[3] = t2;
  stress_mu[4] = f_2_2*2.0-2.0;
  stress_mu[5] = t4;
  stress_mu[6] = t3;
  stress_mu[7] = t4;
  stress_mu[8] = f_3_3*2.0-2.0;

