double t4 = r[0]*r[0];
double t5 = r[1]*r[1];
double t6 = r[2]*r[2];
double t7 = t4+t5+t6;
double t8 = sqrt(t7);
double t9 = cos(t8);
double t10 = 1.0/t7;
double t11 = sin(t8);
double t12 = 1.0/pow(t7,3.0/2.0);
double t13 = t9-1.0;
double t14 = 1.0/(t7*t7);
double t15 = 1.0/sqrt(t7);
double t16 = t5*t9;
double t17 = t6*t9;
double t18 = t4+t16+t17;
double t19 = r[1]*r[2]*t11*t12;
double t20 = t11*t15;
double t21 = r[0]*r[1]*r[2]*t11*t12;
double t22 = r[0]*r[1]*r[2]*t13*t14*2.0;
double t23 = r[0]*r[2]*t11*t12;
double t24 = r[1]*t4*t11*t12;
double t25 = r[1]*t4*t13*t14*2.0;
double t26 = r[0]*t6*t11*t15;
double t27 = r[1]*r[2]*t9*t10;
double t28 = r[0]*t5*t11*t12;
double t29 = r[0]*t5*t13*t14*2.0;
double t30 = r[0]*r[1]*t9*t10;
double t31 = r[1]*t6*t11*t15;
double t32 = t4*t9;
double t33 = t5+t17+t32;
double t34 = r[0]*r[2]*t9*t10;
double t35 = t6*t11*t12;
double t36 = r[2]*t6*t11*t15;
double t37 = r[0]*r[1]*t11*t12;
double t38 = r[2]*t4*t11*t12;
double t39 = r[2]*t4*t13*t14*2.0;
double t40 = t4*t11*t12;
double t41 = r[0]*t4*t11*t15;
double t42 = r[0]*t5*t11*t15;
double t43 = r[2]*t5*t11*t12;
double t44 = r[2]*t5*t13*t14*2.0;
double t45 = t5*t9*t10;
double t46 = r[1]*t5*t11*t15;
double t47 = r[1]*t4*t11*t15;
double t48 = t6+t16+t32;
double t49 = r[0]*t6*t11*t12;
double t50 = r[0]*t6*t13*t14*2.0;
double t51 = r[1]*t6*t11*t12;
double t52 = r[1]*t6*t13*t14*2.0;
double t53 = r[2]*t4*t11*t15;
double t54 = r[2]*t5*t11*t15;
dR(0,0) = t[1]*(t23+t24+t25-r[1]*t10*t13-r[0]*r[2]*t9*t10)+t[2]*(t30+t38+t39-r[2]*t10*t13-r[0]*r[1]*t11*t12)-t[0]*t10*(r[0]*-2.0+t26+t42)-r[0]*t[0]*t14*t18*2.0;
dR(0,1) = t[1]*(t19+t28+t29-r[0]*t10*t13-r[1]*r[2]*t9*t10)+t[2]*(t20+t21+t22+t45-t5*t11*t12)-t[0]*t10*(t31+t46-r[1]*t9*2.0)-r[1]*t[0]*t14*t18*2.0;
dR(0,2) = t[2]*(-t19+t27+t49+t50-r[0]*t10*t13)+t[1]*(-t20+t21+t22+t35-t6*t9*t10)-t[0]*t10*(t36+t54-r[2]*t9*2.0)-r[2]*t[0]*t14*t18*2.0;
dR(1,0) = t[0]*(-t23+t24+t25+t34-r[1]*t10*t13)+t[2]*(-t20+t21+t22+t40-t4*t9*t10)-t[1]*t10*(t26+t41-r[0]*t9*2.0)-r[0]*t[1]*t14*t33*2.0;
dR(1,1) = t[0]*(-t19+t27+t28+t29-r[0]*t10*t13)+t[2]*(-t30+t37+t43+t44-r[2]*t10*t13)-t[1]*t10*(r[1]*-2.0+t31+t47)-r[1]*t[1]*t14*t33*2.0;
dR(1,2) = t[2]*(t23-t34+t51+t52-r[1]*t10*t13)+t[0]*(t20+t21+t22-t35+t6*t9*t10)-t[1]*t10*(t36+t53-r[2]*t9*2.0)-r[2]*t[1]*t14*t33*2.0;
dR(2,0) = t[0]*(-t30+t37+t38+t39-r[2]*t10*t13)+t[1]*(t20+t21+t22-t40+t4*t9*t10)-t[2]*t10*(t41+t42-r[0]*t9*2.0)-r[0]*t[2]*t14*t48*2.0;
dR(2,1) = t[1]*(t30-t37+t43+t44-r[2]*t10*t13)+t[0]*(-t20+t21+t22-t45+t5*t11*t12)-t[2]*t10*(t46+t47-r[1]*t9*2.0)-r[1]*t[2]*t14*t48*2.0;
dR(2,2) = t[0]*(t19-t27+t49+t50-r[0]*t10*t13)+t[1]*(-t23+t34+t51+t52-r[1]*t10*t13)-t[2]*t10*(r[2]*-2.0+t53+t54)-r[2]*t[2]*t14*t48*2.0;