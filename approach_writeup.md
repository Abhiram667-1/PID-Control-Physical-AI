# Approach and reflection

## How I approached the task

I started with the system in the assignment: a 1 kg mass, a spring with stiffness 20 N/m and a damper with coefficient 10 N·s/m. I wrote the equation of motion and used it to update the block's position and velocity in small time steps.

I then wrote a discrete PID controller in Python. I kept the controller separate from the plant simulation so I could change the gains and compare P, PI, PD and PID without changing the model itself. NumPy handles the calculations and Matplotlib creates the graphs.

## What I observed

The P-only results matched the force-balance calculation. The spring needs a constant 20 N to hold the block at 1 m. A P controller can only produce that force while some position error remains, which is why the P-only runs stop short.

Adding integral action reduced the final error because the controller accumulated the offset and built up the force needed to balance the spring. Derivative action changed the transient response, but it did not remove the steady-state offset on its own.

The additional tests show why a practical controller needs more than the basic PID equation. A force limit can cause integral windup, differentiation can make sensor noise more visible, a setpoint step can cause derivative kick, and a slow update rate changes the response.

## Limitations

The plant is a simplified linear model. It does not include all the effects that would appear on a real robot, such as motor electrical dynamics, backlash, changing friction, actuator slew limits or communication delays. The anti-windup method is conditional integration; back-calculation would be a useful comparison.

## What I would try next

With more time, I would compare the numerical integration method with a higher-accuracy solver, add actuator rate limits, and test the controller with different masses and spring stiffnesses. I would also compare the simulated response with measurements from a physical motor or linear stage.

## AI assistance

I used an AI tool to help organize the report and review the wording. I reviewed the technical explanations and kept the Python source in the repository so the calculations and figures can be reproduced. I should be able to explain and modify the code independently in the technical discussion.
