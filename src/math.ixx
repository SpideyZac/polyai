module;
#include <cmath>
#include <iostream>

export module math;

export class Vector3 {
  public:
    double x, y, z;

    Vector3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {};

    double lengthSq() const { return x * x + y * y + z * z; }

    double distanceTo(const Vector3 &other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    bool equals(const Vector3 &other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator==(const Vector3 &other) const { return equals(other); }

    friend std::ostream &operator<<(std::ostream &stream, const Vector3 &self) {
        return stream << "(" << self.x << ", " << self.y << ", " << self.z
                      << ")";
    }
};

export class Quaternion {
  public:
    double x, y, z, w;

    Quaternion(double x = 0, double y = 0, double z = 0, double w = 1)
        : x(x), y(y), z(z), w(w) {}

    bool equals(const Quaternion &other) const {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }

    bool operator==(const Quaternion &other) const { return equals(other); }

    friend std::ostream &operator<<(std::ostream &stream,
                                    const Quaternion &self) {
        return stream << self.w << " + " << self.x << "i + " << self.y << "j + "
                      << self.z << "k";
    }
};
