"""
PID control assignment simulations for Modern Monk Labs.
Dependencies: NumPy and Matplotlib only.
Run from repository root: python src/run_all.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "figures"
OUT.mkdir(exist_ok=True)

DT = 0.001
T_END = 10.0
TIME = np.arange(0.0, T_END + DT, DT)

class PID:
    """Discrete PID with derivative-on-measurement, filtering, and anti-windup."""
    def __init__(self, kp, ki=0.0, kd=0.0, dt=DT, output_limits=(-np.inf, np.inf),
                 derivative_filter_tau=0.02, anti_windup=True, derivative_on_measurement=True):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.low, self.high = output_limits
        self.tau = derivative_filter_tau
        self.anti_windup = anti_windup
        self.derivative_on_measurement = derivative_on_measurement
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.d_filtered = 0.0
        self.initialized = False

    def update(self, setpoint, measurement):
        error = setpoint - measurement
        if not self.initialized:
            self.prev_error = error
            self.prev_measurement = measurement
            self.initialized = True

        if self.derivative_on_measurement:
            raw_d = -(measurement - self.prev_measurement) / self.dt
        else:
            raw_d = (error - self.prev_error) / self.dt
        alpha = self.tau / (self.tau + self.dt) if self.tau > 0 else 0.0
        self.d_filtered = alpha * self.d_filtered + (1.0 - alpha) * raw_d

        candidate_integral = self.integral + error * self.dt
        p = self.kp * error
        d = self.kd * self.d_filtered
        unsaturated = p + self.ki * candidate_integral + d
        saturated = np.clip(unsaturated, self.low, self.high)

        # Conditional integration: stop integrating when saturation would worsen.
        pushing_high = unsaturated > self.high and error > 0
        pushing_low = unsaturated < self.low and error < 0
        if not (self.anti_windup and (pushing_high or pushing_low)):
            self.integral = candidate_integral

        output = np.clip(p + self.ki * self.integral + d, self.low, self.high)
        self.prev_error = error
        self.prev_measurement = measurement
        return output

def simulate_spring_mass_damper(kp, ki=0.0, kd=0.0, force_limit=np.inf,
                                noise_std=0.0, dt=DT, t_end=T_END,
                                setpoint=1.0, derivative_on_measurement=True,
                                anti_windup=True, sample_every=1, derivative_filter_tau=0.02):
    """m*xdd + c*xd + k*x = u; explicit semi-implicit Euler integration."""
    m, c, k = 1.0, 10.0, 20.0
    t = np.arange(0, t_end + dt, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    u = np.zeros_like(t)
    measured = np.zeros_like(t)
    pid = PID(kp, ki, kd, dt, (-force_limit, force_limit),
              derivative_on_measurement=derivative_on_measurement,
              anti_windup=anti_windup, derivative_filter_tau=derivative_filter_tau)
    rng = np.random.default_rng(7)
    for i in range(1, len(t)):
        measured[i-1] = x[i-1] + rng.uniform(-noise_std, noise_std)
        u[i] = pid.update(setpoint, measured[i-1])
        a = (u[i] - c*v[i-1] - k*x[i-1]) / m
        v[i] = v[i-1] + a*dt
        x[i] = x[i-1] + v[i]*dt
    measured[-1] = x[-1] + rng.uniform(-noise_std, noise_std)
    return t, x, u, measured

def metrics(t, y, target=1.0, band=0.02):
    """Rise time (10-90%), percent overshoot, 2% settling time, final error."""
    # Rise time is measured from 10% to 90% of the achieved final value.
    # This remains meaningful for P/PD responses with a steady-state offset.
    final_value = y[-1]
    y10, y90 = 0.1*final_value, 0.9*final_value
    i10 = np.where(y >= y10)[0]
    i90 = np.where(y >= y90)[0]
    rise = (t[i90[0]] - t[i10[0]]) if len(i10) and len(i90) else np.nan
    overshoot = max(0.0, (np.max(y)-target)/abs(target)*100)
    outside = np.where(np.abs(y-target) > band*abs(target))[0]
    settling = t[outside[-1]+1] if len(outside) and outside[-1]+1 < len(t) else (0.0 if not len(outside) else np.nan)
    final_error = target - y[-1]
    return {"Rise time (s)": rise, "Overshoot (%)": overshoot,
            "Settling time (s)": settling, "Final error (m)": final_error}

def save_plot(filename, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=170)
    plt.close()

def spring_plots():
    # P-only sweep
    plt.figure(figsize=(8,4.6))
    for kp in [5, 20, 50, 100]:
        t,y,_,_ = simulate_spring_mass_damper(kp)
        plt.plot(t,y,label=f"Kp={kp}")
    plt.axhline(1,color="black",ls="--",lw=1,label="Setpoint = 1 m")
    save_plot("p_only_sweep.png","P-only response for four proportional gains","Time (s)","Position (m)")

    # Separate PI and PD plots make the effect of adding each term explicit.
    for name, gains in [("PI", (100,40,0)), ("PD", (100,0,25))]:
        t, y, _, _ = simulate_spring_mass_damper(*gains)
        plt.figure(figsize=(8,4.6))
        plt.plot(t,y,label=f"{name}: Kp={gains[0]}, Ki={gains[1]}, Kd={gains[2]}")
        plt.axhline(1,color="black",ls="--",lw=1,label="Setpoint = 1 m")
        save_plot(f"{name.lower()}_response.png",f"{name} controller step response","Time (s)","Position (m)")

    # Individual P, PI, PD, PID comparison
    configs = [("P",100,0,0),("PI",100,40,0),("PD",100,0,25),("PID",100,40,25)]
    plt.figure(figsize=(8,4.6))
    results = {}
    for name,kp,ki,kd in configs:
        t,y,_,_ = simulate_spring_mass_damper(kp,ki,kd)
        plt.plot(t,y,label=name)
        results[name] = metrics(t,y)
    plt.axhline(1,color="black",ls="--",lw=1,label="Setpoint = 1 m")
    save_plot("controller_comparison.png","P, PI, PD and PID step responses","Time (s)","Position (m)")

    # Annotated response measures on the PID curve
    t,y,_,_ = simulate_spring_mass_damper(100,40,25)
    mm = metrics(t,y)
    plt.figure(figsize=(8,4.6))
    plt.plot(t,y,label="PID response")
    plt.axhline(1,color="black",ls="--",label="Target (1 m)")
    peak_i = int(np.argmax(y))
    plt.scatter([t[peak_i]],[y[peak_i]],zorder=4,label=f"Peak / overshoot {mm['Overshoot (%)']:.1f}%")
    plt.scatter([t[-1]],[y[-1]],zorder=4,label=f"Final error {mm['Final error (m)']:.4f} m")
    plt.axvspan(0,mm["Rise time (s)"],alpha=.12,label=f"Rise-time interval ({mm['Rise time (s)']:.2f} s)")
    plt.axvspan(mm["Settling time (s)"],t[-1],alpha=.08,label=f"Settled by {mm['Settling time (s)']:.2f} s")
    save_plot("response_metrics.png","How step-response performance is measured","Time (s)","Position (m)")

    # Force-limited windup, comparing with and without conditional integration
    plt.figure(figsize=(8,4.6))
    for anti,label in [(False,"No anti-windup"),(True,"Conditional integration")]:
        t,y,u,_ = simulate_spring_mass_damper(80,80,15,force_limit=30,
                                              anti_windup=anti,t_end=12)
        plt.plot(t,y,label=label)
    plt.axhline(1,color="black",ls="--",label="Setpoint")
    save_plot("windup.png","Integral windup with a 30 N actuator limit","Time (s)","Position (m)")

    # Noisy derivative: raw vs filtered D
    plt.figure(figsize=(8,4.6))
    for tau,label in [(0.0,"Unfiltered derivative"),(0.02,"Filtered derivative (tau=0.02 s)")]:
        t,y,u,meas = simulate_spring_mass_damper(100,20,12,noise_std=0.001,
                                                 derivative_on_measurement=True,t_end=5, derivative_filter_tau=tau)
        # Plot control effort, which shows noise amplification
        plt.plot(t,u,label=label)
    save_plot("derivative_noise.png","Derivative action amplifies measurement noise","Time (s)","Control force (N)")

    # Derivative kick comparison: setpoint changes from 0 m to 1 m at t=1 s.
    # Measurement is initially stationary, so derivative-on-error spikes while
    # derivative-on-measurement does not react to the setpoint jump itself.
    plt.figure(figsize=(8,4.6))
    t = np.arange(0, 2.0 + DT, DT)
    for dom,label in [(False,"Derivative on error"),(True,"Derivative on measurement")]:
        pid = PID(0.0, 0.0, 20.0, DT, (-1000, 1000),
                  derivative_on_measurement=dom, derivative_filter_tau=0.01)
        effort = np.zeros_like(t)
        measurement = 0.0
        for i,ti in enumerate(t):
            sp = 0.0 if ti < 1.0 else 1.0
            effort[i] = pid.update(sp, measurement)
        plt.plot(t,effort,label=label)
    plt.axvline(1.0,color="black",ls="--",lw=1,label="Setpoint step")
    save_plot("derivative_kick.png","Derivative kick at a 0-to-1 m setpoint change","Time (s)","Derivative control effort (N)")

    # Slow sampling comparison
    plt.figure(figsize=(8,4.6))
    for dt in [0.001,0.02,0.1,0.25]:
        t,y,_,_ = simulate_spring_mass_damper(100,40,25,dt=dt,t_end=10)
        plt.plot(t,y,marker="." if dt>=.1 else None,markevery=max(1,int(.1/dt)),
                 label=f"Update interval {dt:g} s")
    plt.axhline(1,color="black",ls="--",label="Setpoint")
    save_plot("sampling_time.png","Effect of a slower controller update rate","Time (s)","Position (m)")
    return results

def third_order_plant(kp, ki=0.0, kd=0.0, dt=0.001, t_end=20.0):
    """Simulate G(s)=1/(s+1)^3 using three cascaded first-order states."""
    t = np.arange(0,t_end+dt,dt)
    states = np.zeros((3,len(t)))
    u = np.zeros_like(t)
    # Simple discrete PID with derivative-on-measurement and derivative filter.
    pid = PID(kp,ki,kd,dt,(-100,100),derivative_filter_tau=.01)
    for i in range(1,len(t)):
        u[i] = pid.update(1.0,states[2,i-1])
        states[0,i] = states[0,i-1] + dt*(-states[0,i-1] + u[i])
        states[1,i] = states[1,i-1] + dt*(-states[1,i-1] + states[0,i-1])
        states[2,i] = states[2,i-1] + dt*(-states[2,i-1] + states[1,i-1])
    return t,states[2],u

def zn_plot():
    # For G(s)=1/(s+1)^3, P-only characteristic equation:
    # (s+1)^3 + Ku = 0. Routh boundary gives Ku=8, wu=sqrt(3), Pu=2*pi/sqrt(3).
    ku = 8.0
    pu = 2*np.pi/np.sqrt(3)
    kp, ti, td = .6*ku, .5*pu, .125*pu
    ki, kd = kp/ti, kp*td
    plt.figure(figsize=(8,4.6))
    for name, gains in [("P at Ku=8 (marginal)",(ku,0,0)),
                        ("ZN PID",(kp,ki,kd))]:
        t,y,u = third_order_plant(*gains)
        plt.plot(t,y,label=name)
    plt.axhline(1,color="black",ls="--",label="Setpoint")
    save_plot("ziegler_nichols.png","Ziegler–Nichols ultimate-gain tuning on 1/(s+1)^3","Time (s)","Output")
    print(f"Ku={ku:.3f}, Pu={pu:.3f} s; ZN PID: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")

if __name__ == "__main__":
    results = spring_plots()
    zn_plot()
    print("Step-response metrics (2% settling band):")
    for name, values in results.items():
        print(name, {
            k: (round(float(v), 4) if np.isfinite(v) else "Not settled")
            for k, v in values.items()
        })
    print(f"Graphs saved to {OUT.resolve()}")
