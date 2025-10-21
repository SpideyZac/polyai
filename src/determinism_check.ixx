module;
#include <algorithm>
#include <iostream>
#include <memory>

#include "btBulletDynamicsCommon.h"

export module determinism_check;
import math;

class PhysicsWorld {
  private:
    static constexpr int STEPS_PER_SECOND = 1000;
    static constexpr double TIME_STEP = 1.0 / STEPS_PER_SECOND;

    btDefaultCollisionConfiguration m_collisionConfiguration;
    btCollisionDispatcher m_dispatcher;
    btDbvtBroadphase m_broadphase;
    btSequentialImpulseConstraintSolver m_solver;
    btDiscreteDynamicsWorld m_dynamicsWorld;

    struct GroundInfo {
        btRigidBody *body;
        btCollisionShape *shape;
        bool isActive;
    };
    std::unique_ptr<GroundInfo> m_ground;

    struct MountainInfo {
        btRigidBody *body;
        btCollisionShape *shape;
        btTriangleMesh *triangleMesh;
        Vector3 offset;
        double minimumRadius;
        bool isActive;
    };
    std::unique_ptr<MountainInfo> m_mountains;

  public:
    PhysicsWorld()
        : m_collisionConfiguration(), m_dispatcher(&m_collisionConfiguration),
          m_broadphase(), m_solver(),
          m_dynamicsWorld(&m_dispatcher, &m_broadphase, &m_solver,
                          &m_collisionConfiguration) {
        m_dynamicsWorld.setGravity(btVector3(0, -9.82, 0));
    }

    ~PhysicsWorld() { dispose(); }

    void dispose() {
        if (m_ground && m_ground->isActive) {
            m_dynamicsWorld.removeRigidBody(m_ground->body);
        }
        if (m_ground) {
            delete m_ground->body->getMotionState();
            delete m_ground->body;
            delete m_ground->shape;
            m_ground.reset();
        }
    }

    void createGroundPlane() {
        if (m_ground) {
            return;
        }

        btVector3 normal(0, 1, 0);
        auto shape = new btStaticPlaneShape(normal, 0);
        shape->setMargin(0.01);

        btTransform transform;
        transform.setIdentity();
        auto motionState = new btDefaultMotionState(transform);

        btVector3 localInertia(0, 0, 0);
        shape->calculateLocalInertia(0, localInertia);

        btRigidBody::btRigidBodyConstructionInfo rbInfo(0, motionState, shape,
                                                        localInertia);
        auto body = new btRigidBody(rbInfo);
        body->setFriction(1.0);

        m_ground = std::make_unique<GroundInfo>();
        m_ground->body = body;
        m_ground->shape = shape;
        m_ground->isActive = false;
    }

    void createMountains(const double vertices[], const size_t verticeCount,
                         const Vector3 &offset) {
        if (verticeCount % 9 != 0) {
            std::cerr << "Number of mountain vertices is not dividable by 9\n";
            return;
        }
        if (verticeCount == 0) {
            return;
        }
        if (m_mountains) {
            std::cerr << "Mountains are already initialized\n";
            return;
        }

        double minDistanceSq = std::numeric_limits<double>::max();
        auto triangleMesh = new btTriangleMesh();
        for (int i = 0; i < verticeCount; i += 9) {
            btVector3 vec1(vertices[i], vertices[i + 1], vertices[i + 2]);
            btVector3 vec2(vertices[i + 3], vertices[i + 4], vertices[i + 5]);
            btVector3 vec3(vertices[i + 6], vertices[i + 7], vertices[i + 8]);
            triangleMesh->addTriangle(vec1, vec2, vec3, false);

            double l1 = Vector3(vertices[i], vertices[i + 1], vertices[i + 2])
                            .lengthSq();
            double l2 =
                Vector3(vertices[i + 3], vertices[i + 4], vertices[i + 5])
                    .lengthSq();
            double l3 =
                Vector3(vertices[i + 6], vertices[i + 7], vertices[i + 8])
                    .lengthSq();
            minDistanceSq = std::min({minDistanceSq, l1, l2, l3});
        }
        auto mountainMeshShape = new btBvhTriangleMeshShape(triangleMesh, true);
        mountainMeshShape->setMargin(0.02);
        auto offsetVec = new btVector3(offset.x, offset.y, offset.z);
        btTransform transform;
        transform.setIdentity();
        transform.setOrigin(*offsetVec);

        btVector3 inertiaVec;
        mountainMeshShape->calculateLocalInertia(0, inertiaVec);

        auto motionState = new btDefaultMotionState(transform);

        btRigidBody::btRigidBodyConstructionInfo constructionInfo(
            0, motionState, mountainMeshShape, inertiaVec);
        auto mountainsRigidBody = new btRigidBody(constructionInfo);
        mountainsRigidBody->setFriction(1.0);

        double minDistance = std::sqrt(minDistanceSq);

        m_mountains = std::make_unique<MountainInfo>();

        m_mountains->body = mountainsRigidBody;
        m_mountains->shape = mountainMeshShape;
        m_mountains->triangleMesh = triangleMesh;
        m_mountains->offset = offset;
        m_mountains->minimumRadius = minDistance;
        m_mountains->isActive = false;
    }

    void activePhysicsAt(const Vector3 &position) {
        if (m_ground) {
            if (position.y < 4) {
                if (!m_ground->isActive) {
                    m_dynamicsWorld.addRigidBody(m_ground->body);
                    m_ground->isActive = true;
                }
            } else if (position.y > 5 && m_ground->isActive) {
                m_dynamicsWorld.removeRigidBody(m_ground->body);
                m_ground->isActive = false;
            }
        }
        if (m_mountains) {
            double distance = position.distanceTo(m_mountains->offset);
            if (distance > m_mountains->minimumRadius - 10) {
                if (!m_mountains->isActive) {
                    m_dynamicsWorld.addRigidBody(m_mountains->body);
                    m_mountains->isActive = true;
                }
            } else if (distance < m_mountains->minimumRadius - 20 &&
                       m_mountains->isActive) {
                m_dynamicsWorld.removeRigidBody(m_mountains->body);
                m_mountains->isActive = false;
            }
        }
    }
    void step() { m_dynamicsWorld.stepSimulation(TIME_STEP, 0, TIME_STEP); }
    btDiscreteDynamicsWorld &getWorld() { return m_dynamicsWorld; }
    btCollisionDispatcher &getDispatcher() { return m_dispatcher; }
};

export bool determinismCheck() {
    Vector3 expectedPosition(-0.6827400326728821, 0.11212741583585739,
                             2.6956899166107178);
    Quaternion expectedRotation(-0.615668535232544, 0.03904851898550987,
                                0.7859793305397034, 0.04079177975654602);

    PhysicsWorld physicsWorld;
    physicsWorld.createGroundPlane();

    double mountainVertices[] = {2, 3, 4, 5, 4, 5, 7, 8, 10};
    physicsWorld.createMountains(mountainVertices, 9, Vector3(0, 0, 0));

    physicsWorld.activePhysicsAt(Vector3(0, 0, 0));

    btTransform chassisTransform;
    chassisTransform.setIdentity();
    auto chassisMotionState = btDefaultMotionState(chassisTransform);

    btVector3 chassisInertia(0, 0, 0);
    btVector3 chassisHalfExtents(0.1, 0.1, 0.1);
    auto chassisShape = btBoxShape(chassisHalfExtents);
    chassisShape.calculateLocalInertia(400, chassisInertia);

    btRigidBody::btRigidBodyConstructionInfo chassisRbInfo(
        400, &chassisMotionState, &chassisShape, chassisInertia);
    auto chassisBody = btRigidBody(chassisRbInfo);
    chassisBody.setActivationState(DISABLE_DEACTIVATION);
    physicsWorld.getWorld().addRigidBody(&chassisBody);

    btRaycastVehicle::btVehicleTuning tuning;
    auto vehicleRayCaster = btDefaultVehicleRaycaster(&physicsWorld.getWorld());
    auto vehicle = btRaycastVehicle(tuning, &chassisBody, &vehicleRayCaster);
    vehicle.setCoordinateSystem(0, 1, 2);
    physicsWorld.getWorld().addAction(&vehicle);

    btVector3 wheelDirectionCS0(0, -1, 0);
    btVector3 wheelAxleCS(-1, 0, 0);

    struct WheelInfo {
        const char *name;
        btVector3 connectionPoint;
        bool isFrontWheel;
    };

    WheelInfo wheels[] = {
        {"WheelFL", btVector3(0.627909, 0.27, 1.3478), true},
        {"WheelFR", btVector3(-0.627909, 0.27, 1.3478), true},
        {"WheelBL", btVector3(0.720832, 0.27, -1.52686), false},
        {"WheelBR", btVector3(-0.720832, 0.27, -1.52686), false}};

    for (const auto &wheel : wheels) {
        vehicle.addWheel(wheel.connectionPoint, wheelDirectionCS0, wheelAxleCS,
                         0.12, 0.331, tuning, wheel.isFrontWheel);
    }

    btTransform initialTransform;
    initialTransform.setIdentity();
    chassisBody.setWorldTransform(initialTransform);
    chassisBody.getMotionState()->setWorldTransform(initialTransform);

    vehicle.resetSuspension();
    vehicle.setSteeringValue(0, 0);
    vehicle.setSteeringValue(0, 1);

    btTransform secondBodyTransform;
    secondBodyTransform.setIdentity();
    auto secondBodyMotionState = btDefaultMotionState(secondBodyTransform);

    btVector3 secondBodyInertia(0, 0, 0);
    btVector3 secondBodyHalfExtents(0.1, 0.1, 0.1);
    auto secondBodyShape = btBoxShape(secondBodyHalfExtents);
    secondBodyShape.calculateLocalInertia(100, secondBodyInertia);

    btRigidBody::btRigidBodyConstructionInfo secondBodyRbInfo(
        100, &secondBodyMotionState, &secondBodyShape, secondBodyInertia);
    auto secondBody = btRigidBody(secondBodyRbInfo);
    secondBody.setActivationState(DISABLE_DEACTIVATION);
    physicsWorld.getWorld().addRigidBody(&secondBody);

    const float engineForce = 100000.0;
    vehicle.applyEngineForce(engineForce, 2);
    vehicle.applyEngineForce(engineForce, 3);

    for (int i = 0; i < 999; i++) {
        physicsWorld.step();
        btTransform chassisWorldTransform;
        chassisBody.getMotionState()->getWorldTransform(chassisWorldTransform);
    }

    btTransform finalTransform;
    chassisBody.getMotionState()->getWorldTransform(finalTransform);

    btVector3 finalOrigin = finalTransform.getOrigin();
    btQuaternion finalRotation = finalTransform.getRotation();

    Vector3 actualPosition(finalOrigin.x(), finalOrigin.y(), finalOrigin.z());
    Quaternion actualRotation(finalRotation.x(), finalRotation.y(),
                              finalRotation.z(), finalRotation.w());

    bool positionMatch = expectedPosition == actualPosition;
    bool rotationMatch = expectedRotation == actualRotation;

    bool result = positionMatch && rotationMatch;
    if (!result) {
        std::cout << "Expected Position: " << expectedPosition << '\n';
        std::cout << "Actual Position: " << actualPosition << '\n';
        std::cout << "Expected Rotation: " << expectedRotation << '\n';
        std::cout << "Actual Rotation: " << actualRotation << '\n';
        std::cerr << "Determinism check failed: Simulation\n";
    }

    return result;
}
