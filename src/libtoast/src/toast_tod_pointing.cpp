
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_qarray.hpp>
#include <toast/math_sf.hpp>
#include <toast/tod_pointing.hpp>

#include <sstream>
#include <iostream>


void toast::pointing_matrix_healpix(toast::HealpixPixels const & hpix,
                                    bool nest, double eps, double cal,
                                    std::string const & mode, size_t n,
                                    double const * pdata,
                                    double const * hwpang,
                                    uint8_t const * flags,
                                    int64_t * pixels, double * weights) {
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double nullquat[4] = {0.0, 0.0, 0.0, 1.0};

    double eta = (1.0 - eps) / (1.0 + eps);

    toast::AlignedVector <double> dir(3 * n);
    toast::AlignedVector <double> pin(4 * n);

    if (flags == NULL) {
        std::copy(pdata, pdata + (4 * n), pin.begin());
    } else {
        size_t off;
        for (size_t i = 0; i < n; ++i) {
            off = 4 * i;
            if (flags[i] == 0) {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } else {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }
    }

    toast::qa_rotate_many_one(n, pin.data(), zaxis, dir.data());

    if (nest) {
        hpix.vec2nest(n, dir.data(), pixels);
    } else {
        hpix.vec2ring(n, dir.data(), pixels);
    }

    if (flags != NULL) {
        for (size_t i = 0; i < n; ++i) {
            pixels[i] = (flags[i] == 0) ? pixels[i] : -1;
        }
    }

    if (mode == "I") {
        for (size_t i = 0; i < n; ++i) {
            weights[i] = cal;
        }
    } else if (mode == "IQU") {
        toast::AlignedVector <double> orient(3 * n);
        toast::AlignedVector <double> buf1(n);
        toast::AlignedVector <double> buf2(n);

        toast::qa_rotate_many_one(n, pin.data(), xaxis, orient.data());

        double * bx = buf1.data();
        double * by = buf2.data();

        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            by[i] = orient[off + 0] * dir[off + 1] - orient[off + 1] *
                    dir[off + 0];
            bx[i] = orient[off + 0] * (-dir[off + 2] * dir[off + 0]) +
                    orient[off + 1] * (-dir[off + 2] * dir[off + 1]) +
                    orient[off + 2] * (dir[off + 0] * dir[off + 0] +
                                       dir[off + 1] * dir[off + 1]);
        }

        toast::AlignedVector <double> detang(n);

        // FIXME:  Switch back to fast version after unit tests improved.
        toast::vatan2(n, by, bx, detang.data());

        if (hwpang == NULL) {
            for (size_t i = 0; i < n; ++i) {
                detang[i] *= 2.0;
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                detang[i] += 2.0 * hwpang[i];
                detang[i] *= 2.0;
            }
        }

        double * sinout = buf1.data();
        double * cosout = buf2.data();

        // FIXME:  Switch back to fast version after unit tests pass
        toast::vsincos(n, detang.data(), sinout, cosout);

        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            weights[off + 0] = cal;
            weights[off + 1] = cosout[i] * eta * cal;
            weights[off + 2] = sinout[i] * eta * cal;
        }
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "unknown healpix pointing matrix mode \"" << mode << "\"";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    return;
}

void toast::pointing_matrix_healpix_hwp(toast::HealpixPixels const & hpix,
                                    bool nest, double eps, double cal,
                                    std::string const & mode, size_t n,
                                    double const * pdata,
                                    double const * hwpang, double const * hwpparams,
                                    uint8_t const * flags,
                                    int64_t * pixels, double * weights) {
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double nullquat[4] = {0.0, 0.0, 0.0, 1.0};

    double eta = (1.0 - eps) / (1.0 + eps);

    toast::AlignedVector <double> dir(3 * n);
    toast::AlignedVector <double> pin(4 * n);

    if (flags == NULL) {
        std::copy(pdata, pdata + (4 * n), pin.begin());
    } else {
        size_t off;
        for (size_t i = 0; i < n; ++i) {
            off = 4 * i;
            if (flags[i] == 0) {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } else {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }
    }

    toast::qa_rotate_many_one(n, pin.data(), zaxis, dir.data());

    if (nest) {
        hpix.vec2nest(n, dir.data(), pixels);
    } else {
        hpix.vec2ring(n, dir.data(), pixels);
    }

    if (flags != NULL) {
        for (size_t i = 0; i < n; ++i) {
            pixels[i] = (flags[i] == 0) ? pixels[i] : -1;
        }
    }

    if (mode == "I") {
        for (size_t i = 0; i < n; ++i) {
            weights[i] = cal;
        }
    } else if (mode == "IQU") {
        toast::AlignedVector <double> orient(3 * n);
        toast::AlignedVector <double> buf1(n);
        toast::AlignedVector <double> buf2(n);

        toast::qa_rotate_many_one(n, pin.data(), xaxis, orient.data());

        double * bx = buf1.data();
        double * by = buf2.data();

        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            by[i] = orient[off + 0] * dir[off + 1] - orient[off + 1] *
                    dir[off + 0];
            bx[i] = orient[off + 0] * (-dir[off + 2] * dir[off + 0]) +
                    orient[off + 1] * (-dir[off + 2] * dir[off + 1]) +
                    orient[off + 2] * (dir[off + 0] * dir[off + 0] +
                                       dir[off + 1] * dir[off + 1]);
        }
		
		int hwpparamssize = sizeof(hwpparams)/sizeof(hwpparams[0]);
		
		if (hwpparamssize==7) {


        toast::AlignedVector <double> detang(n);
        toast::AlignedVector <double> hwpang2f(n);
        toast::AlignedVector <double> hwpang4f(n);

        // FIXME:  Switch back to fast version after unit tests improved.
        toast::vatan2(n, by, bx, detang.data());

        for (size_t i = 0; i < n; ++i) {
            detang[i] *= 2.0;
        }
        for (size_t i = 0; i < n; ++i) {
            hwpang2f[i] = 2.0*hwpang[i];
        }
        for (size_t i = 0; i < n; ++i) {
            hwpang4f[i] *= 4.0*hwpang[i];
        }
		
		
        toast::AlignedVector <double> buf3(n);
        toast::AlignedVector <double> buf4(n);		
        toast::AlignedVector <double> buf5(n);
        toast::AlignedVector <double> buf6(n);

        double * detsinout = buf1.data();
        double * detcosout = buf2.data();
        double * f2sinout = buf3.data();
        double * f2cosout = buf4.data();
        double * f4sinout = buf5.data();
        double * f4cosout = buf6.data();

        // FIXME:  Switch back to fast version after unit tests pass
        toast::vsincos(n, detang.data(), detsinout, detcosout);
        toast::vsincos(n, hwpang2f.data(), f2sinout, f2cosout);
        toast::vsincos(n, hwpang4f.data(), f4sinout, f4cosout);
		
		double h1=hwpparams[0];
		double h2=hwpparams[1];
		double zeta1=hwpparams[2];
		double zeta2=hwpparams[3];
		double chi1=hwpparams[4];
		double chi2=hwpparams[5];
		double beta=hwpparams[6];
			
		double hII=.5*((1+h1)*(1+h1)+(1+h2)*(1+h2)+zeta1*zeta1+zeta2*zeta2);
		double hQIc2=.5*((1+h1)*(1+h1)-(1+h2)*(1+h2)+zeta1*zeta1-zeta2*zeta2);
		double hQIs2=zeta1*(1+h2)*cos(chi1-beta)-zeta2*(1+h1)*cos(chi2);
		double hUIc2=zeta2*(+h1)*cos(chi2)-zeta1*(1+h2)*cos(chi1-beta);
		double hUIs2=.5*((1+h1)*(1+h1)-(1+h2)*(1+h2)+zeta1*zeta1-zeta2*zeta2);
		double hIQc2=.5*((1+h1)*(1+h1)-(1+h2)*(1+h2)-zeta1*zeta1+zeta2*zeta2);
		double hIQs2=-1.*(zeta1*(1+h1)*cos(chi2)-zeta2*(1+h2)*cos(chi2-beta));
		double hQQ=.25*((1+h1)*(1+h1)+2*(1+h1)*(1+h2)*cos(beta)+(1+h2)*(1+h2)-zeta1*zeta1+2*zeta1*zeta2*cos(chi1-chi2)-zeta2*zeta2);
		double hQQs4=-.5*(zeta1*(1+h1)*cos(chi1)+zeta1*(1+h2)*cos(chi1-beta)+zeta2*(1+h1)*cos(chi2)+zeta2*(1+h2)*cos(chi2-beta));
		double hQQc4=.25*((1+h1)*(1+h1)+2*(1 +h1)*(1 +h2)*cos(beta)+(1+h2)*(1+h2)-zeta1*zeta1-2*zeta1*zeta2*cos(chi1-chi2)-zeta2*zeta2);
		double hUQc2=-.5*(zeta1*(1+h1)*cos(chi1)-zeta2*(1+h1)*cos(chi2)+zeta1*(1+h2)*cos(beta-chi1)-zeta2*(1+h2)*cos(beta-chi2));
		double hUQs2=.5*(zeta1*zeta1-zeta2*zeta2);
		double hUQs4=.25*((1+h1)*(1+h1)+2*(1+h1)*(1+h2)*cos(beta)+(1+h2)*(1+h2)-zeta1*zeta1-2*zeta1*zeta2*cos(chi1-chi2)-zeta2*zeta2);
		double hUQc4=.5*(zeta1*(1+h1)*cos(chi1)+zeta1*(1+h2)+cos(chi1-beta)+zeta2*(1+h1)*cos(chi2)+zeta2*(1+h2)*cos(chi1-beta));
		double hIUc2=zeta1*(1+h1)*cos(chi1)-zeta2*(1+h2)*cos(chi2-beta);
		double hIUs2=.5*((1+h1)*(1+h1)-(1+h2)*(1+h2)-zeta1*zeta1+zeta2*zeta2);
		double hQUc2=.5*(zeta1*(1+h1)*cos(chi1)-zeta2*(1+h1)*cos(chi2)+zeta1*(1+h2)*cos(beta-chi1)-zeta2*(1+h2)*cos(beta-chi2));
		double hQUs2=-.5*(zeta1*zeta1-zeta2*zeta2);
		double hQUs4=.25*((1+h1)*(1+h1)+2*(1+h1)*(1+h2)*cos(beta)+(1+h2)*(1+h2)-zeta1*zeta1-2*zeta1*zeta2*cos(chi1-chi2)-zeta2*zeta2);
		double hQUc4=.5*(zeta1*(1+h1)*cos(chi1)+zeta1*(1+h2)+cos(chi1-beta)+zeta2*(1+h1)*cos(chi2)+zeta2*(1+h2)*cos(chi1-beta));
		double hUU=.25*((1+h1)*(1+h1)+2*(1+h1)*(1+h2)*cos(beta)+(1+h2)*(1+h2)-zeta1*zeta1+2*zeta1*zeta2*cos(chi1-chi2)-zeta2*zeta2);
		double hUUs4=.5*(zeta1*(1+h1)*cos(chi1)+zeta1*(1+h2)*cos(chi1-beta)+zeta2*(1+h1)*cos(chi2)+zeta2*(1+h2)*cos(chi2-beta));
		double hUUc4=-.25*((1+h1)*(1+h1)+2*(1 +h1)*(1 +h2)*cos(beta)+(1+h2)*(1+h2)-zeta1*zeta1-2*zeta1*zeta2*cos(chi1-chi2)-zeta2*zeta2);
// 		double hIV=.5*(-zeta1*(1+h1)*sin(chi1)-zeta2*(1+h2)*sin(chi2)-zeta1*(1+h2)*sin(beta-chi1)-zeta2*(1+h2)*sin(chi2-beta));
// 		double hIVc2=.5*(-zeta1*(1+h1)*sin(chi1)+zeta2*(1+h1)*sin(chi2)+zeta1*(1+h2)*sin(beta-chi1)-zeta2*(1+h2)*sin(beta-chi2));
// 		double hIVs2=zeta1*zeta2*sin(chi1-chi2);
// 		double hQVc2=-1.*(zeta1*(1+h1)*sin(chi1)+zeta2*(1+h2)*cos(beta-chi2));
// 		double hQVs2=-1.*(1+h1)*(1+h2)*sin(beta)+zeta1*zeta2*sin(chi1-chi2);
// 		double hUVs2=zeta1*(1+h2)*sin(beta-chi1)+zeta2*(1+h1)*cos(chi2);
// 		double hUVc2=(1+h1)*(1+h2)*sin(beta)-zeta1*zeta2*sin(chi1-chi2);

		
// 		HIQ=hIQc2*f2cosout[i]+hIQs2*f2sinout[i]
// 		HQQ=hQQ+hQQs4*f4sinout[i]+hQQc4*f4cosout[i]
// 		HUQ=hUQc2*f2cosout[i]+hUQs2*f2sinout[i]+hUQs4*f4sinout[i]+hUQc4*f4cosout[i]
// 		HIU=hIUc2*f2cosout[i]+hIUs2*f2sinout[i]
// 		HQU=hQUc2*f2cosout[i]+hQUs2*f2sinout[i]+hUQs4*f4sinout[i]+hUQc4*f4cosout[i]
// 		HUU=hUU+hUUs4*f4sinout[i]+hUUc4*f4cosout[i]
// 		HIV=hIVc2*f2cosout[i]+hIVs2*f2sinout[i]+hIV
// 		HQV=hQVc2*f2cosout[i]+hQVs2*f2sinout[i]
// 		HUV=hUVc2*f2cosout[i]+hUVs2*f2sinout[i]

        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            weights[off + 0] = cal * .5*(hII+(hQIc2*f2cosout[i]+hQIs2*f2sinout[i])*detcosout[i]+(hUIc2*f2cosout[i]+hUIs2*f2sinout[i])*detsinout[i]); 
            weights[off + 1] = eta * cal * .5*((hIQc2*f2cosout[i]+hIQs2*f2sinout[i])+(hQQ+hQQs4*f4sinout[i]+hQQc4*f4cosout[i])*detcosout[i]+(hUQc2*f2cosout[i]+hUQs2*f2sinout[i]+hUQs4*f4sinout[i]+hUQc4*f4cosout[i])*detsinout[i]);
            weights[off + 2] = eta * cal * .5*((hIUc2*f2cosout[i]+hIUs2*f2sinout[i])+(hQUc2*f2cosout[i]+hQUs2*f2sinout[i]+hUQs4*f4sinout[i]+hUQc4*f4cosout[i])*detcosout[i]+(hUU+hUUs4*f4sinout[i]+hUUc4*f4cosout[i])*detsinout[i]);
            // weights[off + 3] = eta * cal * .5*((hIVc2*f2cosout[i]+hIVs2*f2sinout[i]+hIV)+(hQVc2*f2cosout[i]+hQVs2*f2sinout[i])*detcosout[i]+(hUVc2*f2cosout[i]+hUVs2*f2sinout[i])*detsinout[i]);
        }
		}
		else {
		
        toast::AlignedVector <double> detang(n);
        toast::AlignedVector <double> hwpang2f(n);

        // FIXME:  Switch back to fast version after unit tests improved.
        toast::vatan2(n, by, bx, detang.data());

        for (size_t i = 0; i < n; ++i) {
            detang[i] *= 2.0;
        }
        for (size_t i = 0; i < n; ++i) {
            hwpang2f[i] = 2.0*hwpang[i];
        }
		
		
        toast::AlignedVector <double> buf3(n);
        toast::AlignedVector <double> buf4(n);	

        double * detsinout = buf1.data();
        double * detcosout = buf2.data();
        double * f2sinout = buf3.data();
        double * f2cosout = buf4.data();

        // FIXME:  Switch back to fast version after unit tests pass
        toast::vsincos(n, detang.data(), detsinout, detcosout);
        toast::vsincos(n, hwpang2f.data(), f2sinout, f2cosout);

		
		double MII=hwpparams[0]; 
		double MIQ=hwpparams[1]; 
		double MIU=hwpparams[2]; 
		double MIV=hwpparams[3]; 
		double MQI=hwpparams[4]; 
		double MQQ=hwpparams[5]; 
		double MQU=hwpparams[6]; 
		double MQV=hwpparams[7]; 		
		double MUI=hwpparams[8]; 
		double MUQ=hwpparams[9]; 
		double MUU=hwpparams[10];
		double MUV=hwpparams[11];
		double MVI=hwpparams[12];
		double MVQ=hwpparams[13];
		double MVU=hwpparams[14];
		double MVV=hwpparams[15];
			
			
        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            weights[off + 0] = cal * .5*(MII + MUI*(-1.*f2sinout[i]*detcosout[i] + f2cosout[i]*detsinout[i]) + MQI*(f2cosout[i]*detcosout[i] + f2sinout[i]*detsinout[i])); 
            weights[off + 1] = eta * cal * .5*(f2cosout[i]*(MIQ + MUQ*(-1.*f2sinout[i]*detcosout[i] + f2cosout[i]*detsinout[i]) + MQQ*(f2cosout[i]*detcosout[i] + f2sinout[i]*detsinout[i])) - f2sinout[i]*(MIU + MUU*(-1.*f2sinout[i]*detcosout[i] + f2cosout[i]*detsinout[i]) + MQU*(f2cosout[i]*detcosout[i] + f2sinout[i]*detsinout[i])));
            weights[off + 2] = eta * cal * .5*(f2sinout[i]*(MIQ + MUQ*(-1.*f2sinout[i]*detcosout[i] + f2cosout[i]*detsinout[i]) + MQQ*(f2cosout[i]*detcosout[i] + f2sinout[i]*detsinout[i])) + f2cosout[i]*(MIU + MUU*(-1.*f2sinout[i]*detcosout[i] + f2cosout[i]*detsinout[i]) + MQU*(f2cosout[i]*detcosout[i] + f2sinout[i]*detsinout[i])));
            // weights[off + 3] = eta * cal * .5*(MIV + MUV*(-f2sinout[i]*detcosout[i] + f2cosout[i]*detsinout[i]) + MQV*(f2cosout[i]*detcosout[i] + f2sinout[i]*detsinout[i])));
        }
		}
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "unknown healpix pointing matrix mode \"" << mode << "\"";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    return;
}
