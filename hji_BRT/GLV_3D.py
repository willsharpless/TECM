import heterocl as hcl

class gLV_3D:
    def __init__(self, r, A, x, 
                                uMin = [-0.05,-0.05,-0.05], 
                                uMax = [ 0.05, 0.05, 0.05], 
                                dMin = [-0.025,-0.025,-0.025], 
                                dMax = [ 0.025, 0.025, 0.025], 
                                uMode="max", dMode="min"):
        # u max/d min corresponds to capture default in this setup

        self.x = x
        self.r = r
        self.A = A
        self.uMax = uMax
        self.uMin = uMin
        self.dMax = dMax
        self.dMin = dMin
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
                :param  spat_deriv: tuple of spatial derivative in all dimensions
                        state: x1, x2, x3
                        t: time
                :return: a tuple of optimal disturbances
        """
        opt_u1 = hcl.scalar(self.uMax[0], "opt_u1")
        opt_u2 = hcl.scalar(self.uMax[1], "opt_u2")
        opt_u3 = hcl.scalar(self.uMax[2], "opt_u3")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[0] > 0):
                opt_u1[0] = self.uMin[0]
            with hcl.if_(spat_deriv[1] > 0):
                opt_u2[0] = self.uMin[1]
            with hcl.if_(spat_deriv[2] > 0):
                opt_u3[0] = self.uMin[2]
        else:
            with hcl.if_(spat_deriv[0] < 0):
                opt_u1[0] = self.uMin[0]
            with hcl.if_(spat_deriv[1] < 0):
                opt_u2[0] = self.uMin[1]
            with hcl.if_(spat_deriv[2] < 0):
                opt_u3[0] = self.uMin[2]

        return (opt_u1[0] ,opt_u2[0], opt_u3[0])

    def optDstb(self, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")

        if self.dMode == "max":
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMin[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMax[2]
            with hcl.elif_(spat_deriv[2] < 0):
                d3[0] = self.dMin[2]
        else:
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = self.dMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMax[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMin[2]
            with hcl.elif_(spat_deriv[2] < 0):
                d3[0] = self.dMax[2]

        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")

        # not sure how matmul works with hcl
        int1 = self.A[0][0]*state[0] + self.A[0][1]*state[1] + self.A[0][2]*state[2]
        int2 = self.A[1][0]*state[0] + self.A[1][1]*state[1] + self.A[1][2]*state[2]
        int3 = self.A[2][0]*state[0] + self.A[2][1]*state[1] + self.A[2][2]*state[2]

        x1_dot[0] = state[0]*(self.r[0] + int1) + uOpt[0] + dOpt[0]
        x2_dot[0] = state[1]*(self.r[1] + int2) + uOpt[1] + dOpt[1]
        x3_dot[0] = state[2]*(self.r[2] + int3) + uOpt[2] + dOpt[2]

        return (x1_dot[0], x2_dot[0], x3_dot[0])