import { vec3 } from 'gl-matrix'
import { Point3 } from '../../../types'

type Sphere = {
  center: Point3 | vec3
  radius: number
}

/**
 * Checks if a point is inside a sphere. Note: this is similar to the
 * `pointInEllipse` function, but since we don't need checks for the
 * ellipse's rotation in different views, we can use a simpler equation
 * which would be faster (no if statements).
 *
 * @param sphere Sphere object with center and radius
 * @param pointLPS the point to check in world coordinates
 * @returns boolean
 */
export default function pointInSphere(
  sphere: Sphere,
  pointLPS: Point3
): boolean {
  const { center, radius } = sphere
  const [x, y, z] = pointLPS
  const [x0, y0, z0] = center

  return (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 <= radius ** 2
}