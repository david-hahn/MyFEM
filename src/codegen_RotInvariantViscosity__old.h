double t23 = dv_1_1*f_1_1;
double t24 = dv_2_1*f_2_1;
double t25 = dv_3_1*f_3_1;
double t2 = t23+t24+t25;
double t3 = dv_1_2*f_1_2;
double t30 = dv_2_2*f_2_2;
double t31 = dv_3_2*f_3_2;
double t4 = t3+t30+t31;
double t5 = dv_1_3*f_1_3;
double t34 = dv_2_3*f_2_3;
double t35 = dv_3_3*f_3_3;
double t6 = t5+t34+t35;
double t11 = dv_1_1*f_1_2;
double t12 = dv_1_2*f_1_1;
double t13 = dv_2_1*f_2_2;
double t14 = dv_2_2*f_2_1;
double t15 = dv_3_1*f_3_2;
double t16 = dv_3_2*f_3_1;
double t7 = t11+t12+t13+t14+t15+t16;
double t17 = dv_1_1*f_1_3;
double t18 = dv_1_3*f_1_1;
double t19 = dv_2_1*f_2_3;
double t20 = dv_2_3*f_2_1;
double t21 = dv_3_1*f_3_3;
double t22 = dv_3_3*f_3_1;
double t8 = t17+t18+t19+t20+t21+t22;
double t40 = dv_1_2*f_1_3;
double t41 = dv_1_3*f_1_2;
double t42 = dv_2_2*f_2_3;
double t43 = dv_2_3*f_2_2;
double t44 = dv_3_2*f_3_3;
double t45 = dv_3_3*f_3_2;
double t9 = t40+t41+t42+t43+t44+t45;
double t10 = pow(2.0,3.0/4.0);
double t26 = dv_2_1*f_2_1*2.0;
double t27 = dv_3_1*f_3_1*2.0;
double t28 = t2*t2;
double t29 = t28*2.0;
double t32 = t4*t4;
double t33 = t32*2.0;
double t36 = t6*t6;
double t37 = t36*2.0;
double t38 = t7*t7;
double t39 = t8*t8;
double t46 = t9*t9;
double t47 = t29+t33+t37+t38+t39+t46;
double t48 = 1.0/pow(t47,1.0/4.0);
double t49 = dv_1_1*f_1_1*2.0;
double t50 = t26+t27+t49;
double t51 = f_1_1*nu*t50;
double t52 = f_1_2*nu*t7;
double t53 = f_1_3*nu*t8;
double t54 = t51+t52+t53;
double t55 = 1.0/pow(t47,5.0/4.0);
double t56 = dv_1_2*t7*2.0;
double t57 = dv_1_3*t8*2.0;
double t58 = dv_1_1*t2*4.0;
double t59 = t56+t57+t58;
double t60 = dv_2_2*f_2_2*2.0;
double t61 = dv_3_2*f_3_2*2.0;
double t62 = dv_1_1*t7*2.0;
double t63 = dv_1_3*t9*2.0;
double t64 = dv_1_2*t4*4.0;
double t65 = t62+t63+t64;
double t66 = dv_1_2*f_1_2*2.0;
double t67 = t60+t61+t66;
double t68 = f_1_2*nu*t67;
double t69 = f_1_1*nu*t7;
double t70 = f_1_3*nu*t9;
double t71 = t68+t69+t70;
double t72 = dv_1_1*t8*2.0;
double t73 = dv_1_2*t9*2.0;
double t74 = dv_1_3*t6*4.0;
double t75 = t72+t73+t74;
double t76 = dv_2_2*t7*2.0;
double t77 = dv_2_3*t8*2.0;
double t78 = dv_2_1*t2*4.0;
double t79 = t76+t77+t78;
double t80 = dv_2_1*t7*2.0;
double t81 = dv_2_3*t9*2.0;
double t82 = dv_2_2*t4*4.0;
double t83 = t80+t81+t82;
double t84 = dv_2_3*f_1_3;
double t85 = dv_2_1*t8*2.0;
double t86 = dv_2_2*t9*2.0;
double t87 = dv_2_3*t6*4.0;
double t88 = t85+t86+t87;
double t89 = dv_3_2*t7*2.0;
double t90 = dv_3_3*t8*2.0;
double t91 = dv_3_1*t2*4.0;
double t92 = t89+t90+t91;
double t93 = dv_3_1*t7*2.0;
double t94 = dv_3_3*t9*2.0;
double t95 = dv_3_2*t4*4.0;
double t96 = t93+t94+t95;
double t97 = dv_3_3*f_1_3;
double t98 = dv_3_1*t8*2.0;
double t99 = dv_3_2*t9*2.0;
double t100 = dv_3_3*t6*4.0;
double t101 = t98+t99+t100;
double t102 = dv_1_3*f_1_3*2.0;
double t103 = dv_2_3*f_2_3*2.0;
double t104 = dv_3_3*f_3_3*2.0;
double t105 = t102+t103+t104;
double t106 = f_1_3*nu*t105;
double t107 = f_1_1*nu*t8;
double t108 = f_1_2*nu*t9;
double t109 = t106+t107+t108;
double t110 = dv_2_1*f_1_1;
double t111 = dv_2_2*f_1_2;
double t112 = dv_3_1*f_1_1;
double t113 = dv_3_2*f_1_2;
double t114 = f_2_1*nu*t50;
double t115 = f_2_2*nu*t7;
double t116 = f_2_3*nu*t8;
double t117 = t114+t115+t116;
double t118 = f_2_2*nu*t67;
double t119 = f_2_1*nu*t7;
double t120 = f_2_3*nu*t9;
double t121 = t118+t119+t120;
double t122 = dv_1_3*f_2_3;
double t123 = dv_3_3*f_2_3;
double t124 = f_2_3*nu*t105;
double t125 = f_2_1*nu*t8;
double t126 = f_2_2*nu*t9;
double t127 = t124+t125+t126;
double t128 = dv_1_1*f_2_1;
double t129 = dv_1_2*f_2_2;
double t130 = dv_3_1*f_2_1;
double t131 = dv_3_2*f_2_2;
double t132 = f_3_1*nu*t50;
double t133 = f_3_2*nu*t7;
double t134 = f_3_3*nu*t8;
double t135 = t132+t133+t134;
double t136 = f_3_2*nu*t67;
double t137 = f_3_1*nu*t7;
double t138 = f_3_3*nu*t9;
double t139 = t136+t137+t138;
double t140 = dv_1_3*f_3_3;
double t141 = dv_2_3*f_3_3;
double t142 = f_3_3*nu*t105;
double t143 = f_3_1*nu*t8;
double t144 = f_3_2*nu*t9;
double t145 = t142+t143+t144;
double t146 = dv_1_1*f_3_1;
double t147 = dv_1_2*f_3_2;
double t148 = dv_2_1*f_3_1;
double t149 = dv_2_2*f_3_2;
double t150 = f_1_2*t7*2.0;
double t151 = f_1_3*t8*2.0;
double t152 = f_1_1*t2*4.0;
double t153 = t150+t151+t152;
double t154 = f_1_1*f_1_2*nu*t10*t48*(1.0/2.0);
double t155 = f_1_1*f_1_1;
double t156 = f_1_2*f_1_2;
double t157 = f_1_3*f_1_3;
double t158 = f_1_1*t7*2.0;
double t159 = f_1_3*t9*2.0;
double t160 = f_1_2*t4*4.0;
double t161 = t158+t159+t160;
double t162 = f_1_1*t8*2.0;
double t163 = f_1_2*t9*2.0;
double t164 = f_1_3*t6*4.0;
double t165 = t162+t163+t164;
double t166 = f_2_2*t7*2.0;
double t167 = f_2_3*t8*2.0;
double t168 = f_2_1*t2*4.0;
double t169 = t166+t167+t168;
double t170 = f_2_1*t7*2.0;
double t171 = f_2_3*t9*2.0;
double t172 = f_2_2*t4*4.0;
double t173 = t170+t171+t172;
double t174 = f_1_3*f_2_3;
double t175 = f_2_1*t8*2.0;
double t176 = f_2_2*t9*2.0;
double t177 = f_2_3*t6*4.0;
double t178 = t175+t176+t177;
double t179 = f_3_2*t7*2.0;
double t180 = f_3_3*t8*2.0;
double t181 = f_3_1*t2*4.0;
double t182 = t179+t180+t181;
double t183 = f_3_1*t7*2.0;
double t184 = f_3_3*t9*2.0;
double t185 = f_3_2*t4*4.0;
double t186 = t183+t184+t185;
double t187 = f_1_3*f_3_3;
double t188 = f_3_1*t8*2.0;
double t189 = f_3_2*t9*2.0;
double t190 = f_3_3*t6*4.0;
double t191 = t188+t189+t190;
double t192 = f_1_1*f_1_3*nu*t10*t48*(1.0/2.0);
double t193 = f_1_2*f_1_3*nu*t10*t48*(1.0/2.0);
double t194 = f_1_1*f_2_1;
double t195 = f_1_2*f_2_2;
double t196 = f_1_1*f_3_1;
double t197 = f_1_2*f_3_2;
double t198 = f_1_1*f_2_1*2.0;
double t199 = t174+t195+t198;
double t200 = nu*t10*t48*t199*(1.0/2.0);
double t201 = f_1_1*f_2_2*nu*t10*t48*(1.0/2.0);
double t202 = f_1_1*f_2_3*nu*t10*t48*(1.0/2.0);
double t203 = f_1_2*f_2_1*nu*t10*t48*(1.0/2.0);
double t204 = f_1_2*f_2_2*2.0;
double t205 = t174+t194+t204;
double t206 = nu*t10*t48*t205*(1.0/2.0);
double t207 = f_1_2*f_2_3*nu*t10*t48*(1.0/2.0);
double t208 = f_2_1*f_2_2*nu*t10*t48*(1.0/2.0);
double t209 = f_2_1*f_2_1;
double t210 = f_2_2*f_2_2;
double t211 = f_2_3*f_2_3;
double t212 = f_2_3*f_3_3;
double t213 = f_1_3*f_2_1*nu*t10*t48*(1.0/2.0);
double t214 = f_1_3*f_2_2*nu*t10*t48*(1.0/2.0);
double t215 = f_1_3*f_2_3*2.0;
double t216 = t194+t195+t215;
double t217 = nu*t10*t48*t216*(1.0/2.0);
double t218 = f_2_1*f_2_3*nu*t10*t48*(1.0/2.0);
double t219 = f_2_2*f_2_3*nu*t10*t48*(1.0/2.0);
double t220 = f_2_1*f_3_1;
double t221 = f_2_2*f_3_2;
double t222 = f_1_1*f_3_1*2.0;
double t223 = t187+t197+t222;
double t224 = nu*t10*t48*t223*(1.0/2.0);
double t225 = f_1_1*f_3_2*nu*t10*t48*(1.0/2.0);
double t226 = f_1_1*f_3_3*nu*t10*t48*(1.0/2.0);
double t227 = f_2_1*f_3_1*2.0;
double t228 = t212+t221+t227;
double t229 = nu*t10*t48*t228*(1.0/2.0);
double t230 = f_2_1*f_3_2*nu*t10*t48*(1.0/2.0);
double t231 = f_2_1*f_3_3*nu*t10*t48*(1.0/2.0);
double t232 = f_1_2*f_3_1*nu*t10*t48*(1.0/2.0);
double t233 = f_1_2*f_3_2*2.0;
double t234 = t187+t196+t233;
double t235 = nu*t10*t48*t234*(1.0/2.0);
double t236 = f_1_2*f_3_3*nu*t10*t48*(1.0/2.0);
double t237 = f_2_2*f_3_1*nu*t10*t48*(1.0/2.0);
double t238 = f_2_2*f_3_2*2.0;
double t239 = t212+t220+t238;
double t240 = nu*t10*t48*t239*(1.0/2.0);
double t241 = f_2_2*f_3_3*nu*t10*t48*(1.0/2.0);
double t242 = f_3_1*f_3_2*nu*t10*t48*(1.0/2.0);
double t243 = f_3_1*f_3_1;
double t244 = f_3_2*f_3_2;
double t245 = f_3_3*f_3_3;
double t246 = f_1_3*f_3_1*nu*t10*t48*(1.0/2.0);
double t247 = f_1_3*f_3_2*nu*t10*t48*(1.0/2.0);
double t248 = f_1_3*f_3_3*2.0;
double t249 = t196+t197+t248;
double t250 = nu*t10*t48*t249*(1.0/2.0);
double t251 = f_2_3*f_3_1*nu*t10*t48*(1.0/2.0);
double t252 = f_2_3*f_3_2*nu*t10*t48*(1.0/2.0);
double t253 = f_2_3*f_3_3*2.0;
double t254 = t220+t221+t253;
double t255 = nu*t10*t48*t254*(1.0/2.0);
double t256 = f_3_1*f_3_3*nu*t10*t48*(1.0/2.0);
double t257 = f_3_2*f_3_3*nu*t10*t48*(1.0/2.0);
HF(0,0) = nu*t10*t48*(t3+t5+t26+t27+dv_1_1*f_1_1*4.0)*(1.0/2.0)-t10*t54*t55*t59*(1.0/8.0);
HF(0,1) = t10*t54*t55*t65*(-1.0/8.0)+nu*t10*t48*(t12+t13+t14+t15+t16+dv_1_1*f_1_2*2.0)*(1.0/2.0);
HF(0,2) = t10*t54*t55*t75*(-1.0/8.0)+nu*t10*t48*(t18+t19+t20+t21+t22+dv_1_1*f_1_3*2.0)*(1.0/2.0);
HF(0,3) = nu*t10*t48*(t84+t111+dv_2_1*f_1_1*2.0)*(1.0/2.0)-t10*t54*t55*t79*(1.0/8.0);
HF(0,4) = t10*t54*t55*t83*(-1.0/8.0)+dv_2_1*f_1_2*nu*t10*t48*(1.0/2.0);
HF(0,5) = t10*t54*t55*t88*(-1.0/8.0)+dv_2_1*f_1_3*nu*t10*t48*(1.0/2.0);
HF(0,6) = nu*t10*t48*(t97+t113+dv_3_1*f_1_1*2.0)*(1.0/2.0)-t10*t54*t55*t92*(1.0/8.0);
HF(0,7) = t10*t54*t55*t96*(-1.0/8.0)+dv_3_1*f_1_2*nu*t10*t48*(1.0/2.0);
HF(0,8) = t10*t54*t55*t101*(-1.0/8.0)+dv_3_1*f_1_3*nu*t10*t48*(1.0/2.0);
HF(1,0) = t10*t55*t59*t71*(-1.0/8.0)+nu*t10*t48*(t11+t13+t14+t15+t16+dv_1_2*f_1_1*2.0)*(1.0/2.0);
HF(1,1) = nu*t10*t48*(t5+t23+t60+t61+dv_1_2*f_1_2*4.0)*(1.0/2.0)-t10*t55*t65*t71*(1.0/8.0);
HF(1,2) = t10*t55*t71*t75*(-1.0/8.0)+nu*t10*t48*(t41+t42+t43+t44+t45+dv_1_2*f_1_3*2.0)*(1.0/2.0);
HF(1,3) = t10*t55*t71*t79*(-1.0/8.0)+dv_2_2*f_1_1*nu*t10*t48*(1.0/2.0);
HF(1,4) = nu*t10*t48*(t84+t110+dv_2_2*f_1_2*2.0)*(1.0/2.0)-t10*t55*t71*t83*(1.0/8.0);
HF(1,5) = t10*t55*t71*t88*(-1.0/8.0)+dv_2_2*f_1_3*nu*t10*t48*(1.0/2.0);
HF(1,6) = t10*t55*t71*t92*(-1.0/8.0)+dv_3_2*f_1_1*nu*t10*t48*(1.0/2.0);
HF(1,7) = nu*t10*t48*(t97+t112+dv_3_2*f_1_2*2.0)*(1.0/2.0)-t10*t55*t71*t96*(1.0/8.0);
HF(1,8) = t10*t55*t71*t101*(-1.0/8.0)+dv_3_2*f_1_3*nu*t10*t48*(1.0/2.0);
HF(2,0) = t10*t55*t59*t109*(-1.0/8.0)+nu*t10*t48*(t17+t19+t20+t21+t22+dv_1_3*f_1_1*2.0)*(1.0/2.0);
HF(2,1) = t10*t55*t65*t109*(-1.0/8.0)+nu*t10*t48*(t40+t42+t43+t44+t45+dv_1_3*f_1_2*2.0)*(1.0/2.0);
HF(2,2) = nu*t10*t48*(t3+t23+t103+t104+dv_1_3*f_1_3*4.0)*(1.0/2.0)-t10*t55*t75*t109*(1.0/8.0);
HF(2,3) = t10*t55*t79*t109*(-1.0/8.0)+dv_2_3*f_1_1*nu*t10*t48*(1.0/2.0);
HF(2,4) = t10*t55*t83*t109*(-1.0/8.0)+dv_2_3*f_1_2*nu*t10*t48*(1.0/2.0);
HF(2,5) = nu*t10*t48*(t110+t111+dv_2_3*f_1_3*2.0)*(1.0/2.0)-t10*t55*t88*t109*(1.0/8.0);
HF(2,6) = t10*t55*t92*t109*(-1.0/8.0)+dv_3_3*f_1_1*nu*t10*t48*(1.0/2.0);
HF(2,7) = t10*t55*t96*t109*(-1.0/8.0)+dv_3_3*f_1_2*nu*t10*t48*(1.0/2.0);
HF(2,8) = nu*t10*t48*(t112+t113+dv_3_3*f_1_3*2.0)*(1.0/2.0)-t10*t55*t101*t109*(1.0/8.0);
HF(3,0) = nu*t10*t48*(t122+t129+dv_1_1*f_2_1*2.0)*(1.0/2.0)-t10*t55*t59*t117*(1.0/8.0);
HF(3,1) = t10*t55*t65*t117*(-1.0/8.0)+dv_1_1*f_2_2*nu*t10*t48*(1.0/2.0);
HF(3,2) = t10*t55*t75*t117*(-1.0/8.0)+dv_1_1*f_2_3*nu*t10*t48*(1.0/2.0);
HF(3,3) = nu*t10*t48*(t27+t30+t34+t49+dv_2_1*f_2_1*4.0)*(1.0/2.0)-t10*t55*t79*t117*(1.0/8.0);
HF(3,4) = t10*t55*t83*t117*(-1.0/8.0)+nu*t10*t48*(t11+t12+t14+t15+t16+dv_2_1*f_2_2*2.0)*(1.0/2.0);
HF(3,5) = t10*t55*t88*t117*(-1.0/8.0)+nu*t10*t48*(t17+t18+t20+t21+t22+dv_2_1*f_2_3*2.0)*(1.0/2.0);
HF(3,6) = nu*t10*t48*(t123+t131+dv_3_1*f_2_1*2.0)*(1.0/2.0)-t10*t55*t92*t117*(1.0/8.0);
HF(3,7) = t10*t55*t96*t117*(-1.0/8.0)+dv_3_1*f_2_2*nu*t10*t48*(1.0/2.0);
HF(3,8) = t10*t55*t101*t117*(-1.0/8.0)+dv_3_1*f_2_3*nu*t10*t48*(1.0/2.0);
HF(4,0) = t10*t55*t59*t121*(-1.0/8.0)+dv_1_2*f_2_1*nu*t10*t48*(1.0/2.0);
HF(4,1) = nu*t10*t48*(t122+t128+dv_1_2*f_2_2*2.0)*(1.0/2.0)-t10*t55*t65*t121*(1.0/8.0);
HF(4,2) = t10*t55*t75*t121*(-1.0/8.0)+dv_1_2*f_2_3*nu*t10*t48*(1.0/2.0);
HF(4,3) = t10*t55*t79*t121*(-1.0/8.0)+nu*t10*t48*(t11+t12+t13+t15+t16+dv_2_2*f_2_1*2.0)*(1.0/2.0);
HF(4,4) = nu*t10*t48*(t24+t34+t61+t66+dv_2_2*f_2_2*4.0)*(1.0/2.0)-t10*t55*t83*t121*(1.0/8.0);
HF(4,5) = t10*t55*t88*t121*(-1.0/8.0)+nu*t10*t48*(t40+t41+t43+t44+t45+dv_2_2*f_2_3*2.0)*(1.0/2.0);
HF(4,6) = t10*t55*t92*t121*(-1.0/8.0)+dv_3_2*f_2_1*nu*t10*t48*(1.0/2.0);
HF(4,7) = nu*t10*t48*(t123+t130+dv_3_2*f_2_2*2.0)*(1.0/2.0)-t10*t55*t96*t121*(1.0/8.0);
HF(4,8) = t10*t55*t101*t121*(-1.0/8.0)+dv_3_2*f_2_3*nu*t10*t48*(1.0/2.0);
HF(5,0) = t10*t55*t59*t127*(-1.0/8.0)+dv_1_3*f_2_1*nu*t10*t48*(1.0/2.0);
HF(5,1) = t10*t55*t65*t127*(-1.0/8.0)+dv_1_3*f_2_2*nu*t10*t48*(1.0/2.0);
HF(5,2) = nu*t10*t48*(t128+t129+dv_1_3*f_2_3*2.0)*(1.0/2.0)-t10*t55*t75*t127*(1.0/8.0);
HF(5,3) = t10*t55*t79*t127*(-1.0/8.0)+nu*t10*t48*(t17+t18+t19+t21+t22+dv_2_3*f_2_1*2.0)*(1.0/2.0);
HF(5,4) = t10*t55*t83*t127*(-1.0/8.0)+nu*t10*t48*(t40+t41+t42+t44+t45+dv_2_3*f_2_2*2.0)*(1.0/2.0);
HF(5,5) = nu*t10*t48*(t24+t30+t102+t104+dv_2_3*f_2_3*4.0)*(1.0/2.0)-t10*t55*t88*t127*(1.0/8.0);
HF(5,6) = t10*t55*t92*t127*(-1.0/8.0)+dv_3_3*f_2_1*nu*t10*t48*(1.0/2.0);
HF(5,7) = t10*t55*t96*t127*(-1.0/8.0)+dv_3_3*f_2_2*nu*t10*t48*(1.0/2.0);
HF(5,8) = nu*t10*t48*(t130+t131+dv_3_3*f_2_3*2.0)*(1.0/2.0)-t10*t55*t101*t127*(1.0/8.0);
HF(6,0) = nu*t10*t48*(t140+t147+dv_1_1*f_3_1*2.0)*(1.0/2.0)-t10*t55*t59*t135*(1.0/8.0);
HF(6,1) = t10*t55*t65*t135*(-1.0/8.0)+dv_1_1*f_3_2*nu*t10*t48*(1.0/2.0);
HF(6,2) = t10*t55*t75*t135*(-1.0/8.0)+dv_1_1*f_3_3*nu*t10*t48*(1.0/2.0);
HF(6,3) = nu*t10*t48*(t141+t149+dv_2_1*f_3_1*2.0)*(1.0/2.0)-t10*t55*t79*t135*(1.0/8.0);
HF(6,4) = t10*t55*t83*t135*(-1.0/8.0)+dv_2_1*f_3_2*nu*t10*t48*(1.0/2.0);
HF(6,5) = t10*t55*t88*t135*(-1.0/8.0)+dv_2_1*f_3_3*nu*t10*t48*(1.0/2.0);
HF(6,6) = nu*t10*t48*(t26+t31+t35+t49+dv_3_1*f_3_1*4.0)*(1.0/2.0)-t10*t55*t92*t135*(1.0/8.0);
HF(6,7) = t10*t55*t96*t135*(-1.0/8.0)+nu*t10*t48*(t11+t12+t13+t14+t16+dv_3_1*f_3_2*2.0)*(1.0/2.0);
HF(6,8) = t10*t55*t101*t135*(-1.0/8.0)+nu*t10*t48*(t17+t18+t19+t20+t22+dv_3_1*f_3_3*2.0)*(1.0/2.0);
HF(7,0) = t10*t55*t59*t139*(-1.0/8.0)+dv_1_2*f_3_1*nu*t10*t48*(1.0/2.0);
HF(7,1) = nu*t10*t48*(t140+t146+dv_1_2*f_3_2*2.0)*(1.0/2.0)-t10*t55*t65*t139*(1.0/8.0);
HF(7,2) = t10*t55*t75*t139*(-1.0/8.0)+dv_1_2*f_3_3*nu*t10*t48*(1.0/2.0);
HF(7,3) = t10*t55*t79*t139*(-1.0/8.0)+dv_2_2*f_3_1*nu*t10*t48*(1.0/2.0);
HF(7,4) = nu*t10*t48*(t141+t148+dv_2_2*f_3_2*2.0)*(1.0/2.0)-t10*t55*t83*t139*(1.0/8.0);
HF(7,5) = t10*t55*t88*t139*(-1.0/8.0)+dv_2_2*f_3_3*nu*t10*t48*(1.0/2.0);
HF(7,6) = t10*t55*t92*t139*(-1.0/8.0)+nu*t10*t48*(t11+t12+t13+t14+t15+dv_3_2*f_3_1*2.0)*(1.0/2.0);
HF(7,7) = nu*t10*t48*(t25+t35+t60+t66+dv_3_2*f_3_2*4.0)*(1.0/2.0)-t10*t55*t96*t139*(1.0/8.0);
HF(7,8) = t10*t55*t101*t139*(-1.0/8.0)+nu*t10*t48*(t40+t41+t42+t43+t45+dv_3_2*f_3_3*2.0)*(1.0/2.0);
HF(8,0) = t10*t55*t59*t145*(-1.0/8.0)+dv_1_3*f_3_1*nu*t10*t48*(1.0/2.0);
HF(8,1) = t10*t55*t65*t145*(-1.0/8.0)+dv_1_3*f_3_2*nu*t10*t48*(1.0/2.0);
HF(8,2) = nu*t10*t48*(t146+t147+dv_1_3*f_3_3*2.0)*(1.0/2.0)-t10*t55*t75*t145*(1.0/8.0);
HF(8,3) = t10*t55*t79*t145*(-1.0/8.0)+dv_2_3*f_3_1*nu*t10*t48*(1.0/2.0);
HF(8,4) = t10*t55*t83*t145*(-1.0/8.0)+dv_2_3*f_3_2*nu*t10*t48*(1.0/2.0);
HF(8,5) = nu*t10*t48*(t148+t149+dv_2_3*f_3_3*2.0)*(1.0/2.0)-t10*t55*t88*t145*(1.0/8.0);
HF(8,6) = t10*t55*t92*t145*(-1.0/8.0)+nu*t10*t48*(t17+t18+t19+t20+t21+dv_3_3*f_3_1*2.0)*(1.0/2.0);
HF(8,7) = t10*t55*t96*t145*(-1.0/8.0)+nu*t10*t48*(t40+t41+t42+t43+t44+dv_3_3*f_3_2*2.0)*(1.0/2.0);
HF(8,8) = nu*t10*t48*(t25+t31+t102+t103+dv_3_3*f_3_3*4.0)*(1.0/2.0)-t10*t55*t101*t145*(1.0/8.0);
 H(0,0) = t10*t54*t55*t153*(-1.0/8.0)+nu*t10*t48*(t155*2.0+t156+t157)*(1.0/2.0);
 H(0,1) = t154-t10*t54*t55*t161*(1.0/8.0);
 H(0,2) = t192-t10*t54*t55*t165*(1.0/8.0);
 H(0,3) = t200-t10*t54*t55*t169*(1.0/8.0);
 H(0,4) = t203-t10*t54*t55*t173*(1.0/8.0);
 H(0,5) = t213-t10*t54*t55*t178*(1.0/8.0);
 H(0,6) = t224-t10*t54*t55*t182*(1.0/8.0);
 H(0,7) = t232-t10*t54*t55*t186*(1.0/8.0);
 H(0,8) = t246-t10*t54*t55*t191*(1.0/8.0);
 H(1,0) = t154-t10*t55*t71*t153*(1.0/8.0);
 H(1,1) = t10*t55*t71*t161*(-1.0/8.0)+nu*t10*t48*(t155+t156*2.0+t157)*(1.0/2.0);
 H(1,2) = t193-t10*t55*t71*t165*(1.0/8.0);
 H(1,3) = t201-t10*t55*t71*t169*(1.0/8.0);
 H(1,4) = t206-t10*t55*t71*t173*(1.0/8.0);
 H(1,5) = t214-t10*t55*t71*t178*(1.0/8.0);
 H(1,6) = t225-t10*t55*t71*t182*(1.0/8.0);
 H(1,7) = t235-t10*t55*t71*t186*(1.0/8.0);
 H(1,8) = t247-t10*t55*t71*t191*(1.0/8.0);
 H(2,0) = t192-t10*t55*t109*t153*(1.0/8.0);
 H(2,1) = t193-t10*t55*t109*t161*(1.0/8.0);
 H(2,2) = t10*t55*t109*t165*(-1.0/8.0)+nu*t10*t48*(t155+t156+t157*2.0)*(1.0/2.0);
 H(2,3) = t202-t10*t55*t109*t169*(1.0/8.0);
 H(2,4) = t207-t10*t55*t109*t173*(1.0/8.0);
 H(2,5) = t217-t10*t55*t109*t178*(1.0/8.0);
 H(2,6) = t226-t10*t55*t109*t182*(1.0/8.0);
 H(2,7) = t236-t10*t55*t109*t186*(1.0/8.0);
 H(2,8) = t250-t10*t55*t109*t191*(1.0/8.0);
 H(3,0) = t200-t10*t55*t117*t153*(1.0/8.0);
 H(3,1) = t201-t10*t55*t117*t161*(1.0/8.0);
 H(3,2) = t202-t10*t55*t117*t165*(1.0/8.0);
 H(3,3) = t10*t55*t117*t169*(-1.0/8.0)+nu*t10*t48*(t209*2.0+t210+t211)*(1.0/2.0);
 H(3,4) = t208-t10*t55*t117*t173*(1.0/8.0);
 H(3,5) = t218-t10*t55*t117*t178*(1.0/8.0);
 H(3,6) = t229-t10*t55*t117*t182*(1.0/8.0);
 H(3,7) = t237-t10*t55*t117*t186*(1.0/8.0);
 H(3,8) = t251-t10*t55*t117*t191*(1.0/8.0);
 H(4,0) = t203-t10*t55*t121*t153*(1.0/8.0);
 H(4,1) = t206-t10*t55*t121*t161*(1.0/8.0);
 H(4,2) = t207-t10*t55*t121*t165*(1.0/8.0);
 H(4,3) = t208-t10*t55*t121*t169*(1.0/8.0);
 H(4,4) = t10*t55*t121*t173*(-1.0/8.0)+nu*t10*t48*(t209+t210*2.0+t211)*(1.0/2.0);
 H(4,5) = t219-t10*t55*t121*t178*(1.0/8.0);
 H(4,6) = t230-t10*t55*t121*t182*(1.0/8.0);
 H(4,7) = t240-t10*t55*t121*t186*(1.0/8.0);
 H(4,8) = t252-t10*t55*t121*t191*(1.0/8.0);
 H(5,0) = t213-t10*t55*t127*t153*(1.0/8.0);
 H(5,1) = t214-t10*t55*t127*t161*(1.0/8.0);
 H(5,2) = t217-t10*t55*t127*t165*(1.0/8.0);
 H(5,3) = t218-t10*t55*t127*t169*(1.0/8.0);
 H(5,4) = t219-t10*t55*t127*t173*(1.0/8.0);
 H(5,5) = t10*t55*t127*t178*(-1.0/8.0)+nu*t10*t48*(t209+t210+t211*2.0)*(1.0/2.0);
 H(5,6) = t231-t10*t55*t127*t182*(1.0/8.0);
 H(5,7) = t241-t10*t55*t127*t186*(1.0/8.0);
 H(5,8) = t255-t10*t55*t127*t191*(1.0/8.0);
 H(6,0) = t224-t10*t55*t135*t153*(1.0/8.0);
 H(6,1) = t225-t10*t55*t135*t161*(1.0/8.0);
 H(6,2) = t226-t10*t55*t135*t165*(1.0/8.0);
 H(6,3) = t229-t10*t55*t135*t169*(1.0/8.0);
 H(6,4) = t230-t10*t55*t135*t173*(1.0/8.0);
 H(6,5) = t231-t10*t55*t135*t178*(1.0/8.0);
 H(6,6) = t10*t55*t135*t182*(-1.0/8.0)+nu*t10*t48*(t243*2.0+t244+t245)*(1.0/2.0);
 H(6,7) = t242-t10*t55*t135*t186*(1.0/8.0);
 H(6,8) = t256-t10*t55*t135*t191*(1.0/8.0);
 H(7,0) = t232-t10*t55*t139*t153*(1.0/8.0);
 H(7,1) = t235-t10*t55*t139*t161*(1.0/8.0);
 H(7,2) = t236-t10*t55*t139*t165*(1.0/8.0);
 H(7,3) = t237-t10*t55*t139*t169*(1.0/8.0);
 H(7,4) = t240-t10*t55*t139*t173*(1.0/8.0);
 H(7,5) = t241-t10*t55*t139*t178*(1.0/8.0);
 H(7,6) = t242-t10*t55*t139*t182*(1.0/8.0);
 H(7,7) = t10*t55*t139*t186*(-1.0/8.0)+nu*t10*t48*(t243+t244*2.0+t245)*(1.0/2.0);
 H(7,8) = t257-t10*t55*t139*t191*(1.0/8.0);
 H(8,0) = t246-t10*t55*t145*t153*(1.0/8.0);
 H(8,1) = t247-t10*t55*t145*t161*(1.0/8.0);
 H(8,2) = t250-t10*t55*t145*t165*(1.0/8.0);
 H(8,3) = t251-t10*t55*t145*t169*(1.0/8.0);
 H(8,4) = t252-t10*t55*t145*t173*(1.0/8.0);
 H(8,5) = t255-t10*t55*t145*t178*(1.0/8.0);
 H(8,6) = t256-t10*t55*t145*t182*(1.0/8.0);
 H(8,7) = t257-t10*t55*t145*t186*(1.0/8.0);
 H(8,8) = t10*t55*t145*t191*(-1.0/8.0)+nu*t10*t48*(t243+t244+t245*2.0)*(1.0/2.0);
stress[0] = t10*t48*t54*(1.0/2.0);
stress[1] = t10*t48*t71*(1.0/2.0);
stress[2] = t10*t48*t109*(1.0/2.0);
stress[3] = t10*t48*t117*(1.0/2.0);
stress[4] = t10*t48*t121*(1.0/2.0);
stress[5] = t10*t48*t127*(1.0/2.0);
stress[6] = t10*t48*t135*(1.0/2.0);
stress[7] = t10*t48*t139*(1.0/2.0);
stress[8] = t10*t48*t145*(1.0/2.0);
