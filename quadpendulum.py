import pygame
from pygame.locals import *
import numpy as np
import sys

class Simulator(object):

    def __init__(self):
        self.display_flags = 0
        self.display_size = (1080, 1000)
        self.screen = None
        self.chassis_pos = np.array([self.display_size[0]/2, self.display_size[1]/2])  
        # pendulum properties
        self.leg_lengths = [100, 100, 100, 100]  
        self.leg_masses = [1, 1, 1, 1]  
        # initial angles
        self.angles = [np.pi/4, np.pi/4, np.pi/4, np.pi/4]  
        # initial velocities
        self.angular_velocities = [0.0, 0.0, 0.0, 0.0]  

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.blue = (0, 5, 255)

    def draw(self):
        self.screen.fill(self.white) 
        current_pos = self.chassis_pos
        for i, angle in enumerate(self.angles):
            next_pos = current_pos + np.array([
                self.leg_lengths[i] * np.sin(angle),
                self.leg_lengths[i] * np.cos(angle)
            ])
            pygame.draw.line(self.screen, self.black, current_pos, next_pos, 5)
            pygame.draw.circle(self.screen, self.blue, next_pos.astype(int), 10) 
            current_pos = next_pos
        pygame.display.flip()  

    def calculate_angular_accelerations(self, angles, angular_velocities):
        theta1, theta2, theta3, theta4 = angles
        omega1, omega2, omega3, omega4 = angular_velocities
        l1, l2, l3, l4 = self.leg_lengths
        m1, m2, m3, m4 = self.leg_masses
        g = 9.81
        cos12 = np.cos(theta1 - theta2)
        cos13 = np.cos(theta1 - theta3)
        cos14 = np.cos(theta1 - theta4)
        cos21 = np.cos(theta2 - theta1)
        cos23 = np.cos(theta2 - theta3)
        cos24 = np.cos(theta2 - theta4)
        cos31 = np.cos(theta3 - theta1)
        cos32 = np.cos(theta3 - theta2)
        cos34 = np.cos(theta3 - theta4)
        cos41 = np.cos(theta4 - theta1)
        cos42 = np.cos(theta4 - theta2)
        cos43 = np.cos(theta4 - theta3)
        sin12 = np.sin(theta1 - theta2)
        sin13 = np.sin(theta1 - theta3)
        sin14 = np.sin(theta1 - theta4)
        sin21 = np.sin(theta2 - theta1)
        sin23 = np.sin(theta2 - theta3)
        sin24 = np.sin(theta2 - theta4)
        sin34 = np.sin(theta3 - theta4)
        sin31 = np.sin(theta3 - theta1)
        sin32 = np.sin(theta3 - theta2)
        sin41 = np.sin(theta4 - theta1)
        sin42 = np.sin(theta4 - theta2)
        sin43 = np.sin(theta4 - theta3)
        M = np.zeros((4, 4))
        C = np.zeros(4)
        M[0, 0] = l1**2 * (m1 + m2 + m3 + m4)
        M[0, 1] = l1 * l2 * (m2 + m3 + m4) * cos12
        M[0, 2] = l1 * l3 * (m3 + m4) * cos13
        M[0, 3] = l1 * l4 * m4 * cos14
        C[0] = l1 * (
        - g * np.sin(theta1) * (m1 + m2 + m3 + m4)
        - l2 * (m2 + m3 + m4) * sin12 * omega2 * omega1
        - l3 * (m3 + m4) * sin13 * omega3 * omega1
        - m4 * l4 * sin14 * omega4 * omega1
        - (m2 + m3 + m4) * l2 * sin21 * (omega1 - omega2) * omega2
        - (m3 + m4) * l3 * sin31 * (omega1 - omega3) * omega3
        - m4 * l4 * sin41 * (omega1 - omega4) * omega4
        )
        M[1, 0] = l2 * l1 * (m2 + m3 + m4) * cos21
        M[1, 1] = l2**2 * (m2 + m3 + m4)
        M[1, 2] = l2 * l3 * (m3 + m4) * cos23
        M[1, 3] = l2 * l4 * m4 * cos24
        C[1] = l2 * (
        - g * np.sin(theta2) * (m2 + m3 + m4)
        - l2 * (m2 + m3 + m4) * sin21 * omega1 * omega2
        - l3 * (m3 + m4) * sin23 * omega3 * omega2
        - m4 * l4 * sin24 * omega4 * omega2
        - (m2 + m3 + m4) * l1 * sin12 * (omega2 - omega1) * omega1
        - (m3 + m4) * l3 * sin32 * (omega2 - omega3) * omega3
        - m4 * l4 * sin42 * (omega2 - omega4) * omega4
        )
        M[2, 0] = l3 * l1 * (m3 + m4) * cos31
        M[2, 1] = l3 * l2 * (m3 + m4) * cos32
        M[2, 2] = l3**2 * (m3 + m4)
        M[2, 3] = l3 * l4 * m4 * cos34
        C[2] = l3 * (
        - g * np.sin(theta3) * (m3 + m4)
        - l1 * (m3 + m4) * sin31 * omega1 * omega3 
        - l2 * (m3 + m4) * sin32 * omega2 * omega3
        - m4 * l4 * sin34 * omega4 * omega3
        - (m3 + m4) * l1 * sin13 * (omega3 - omega1) * omega1
        - (m3 + m4) * l2 * sin23 * (omega3 - omega2) * omega2
        - m4 * l4 * sin43 * (omega3 - omega4) * omega4
        )
        M[3, 0] = l4 * l1 * m4 * cos41
        M[3, 1] = l4 * l2 * m4 * cos42
        M[3, 2] = l4 * l3 * m4 * cos43
        M[3, 3] = l4**2 * m4
        C[3] = l4 * (
        - g * np.sin(theta4) * m4
        - l1 * m4 * sin41 * omega1 * omega4
        - l2 * m4 * sin42 * omega2 * omega4
        - m4 * l3 * sin43 * omega3 * omega4
        - m4 * l1 * sin14 * (omega4 - omega1) * omega1
        - m4 * l2 * sin24 * (omega4 - omega2) * omega2
        - m4 * l3 * sin34 * (omega4 - omega3) * omega3
        )
        domega_dt = np.linalg.solve(M, C)
        return domega_dt
    def main(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.display_size, self.display_flags)
        clock = pygame.time.Clock()
        running = True
        font = pygame.font.Font(None, 16)
        simulate = False
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key in (K_q, K_ESCAPE)):
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_s:
                    simulate = not simulate
            if simulate:
                fps = 40
                dt = 1.0 / (fps)  
                domega_dt = self.calculate_angular_accelerations(self.angles, self.angular_velocities)
                for i in range(4):
                    self.angular_velocities[i] += domega_dt[i] * dt
                    self.angles[i] += self.angular_velocities[i] * dt
            self.draw()
            for i, angle in enumerate(self.angles):
                text = font.render(f'Joint {i+1}: {angle:.2f} rad', True, self.black)
                self.screen.blit(text, (10, 10 + i*20))
            for i, velocity in enumerate(self.angular_velocities):
                text = font.render(f'Joint {i+1}: Velocity: {velocity:.2f} rad/s', True, self.black)
                self.screen.blit(text, (10, 90 + i * 20))
            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    sim = Simulator()
    sim.main()
