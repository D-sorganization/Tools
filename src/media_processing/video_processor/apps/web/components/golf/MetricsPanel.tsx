'use client';

import React from 'react';
import {
  SwingAnalysis,
  BalanceMetrics,
  PlaneMetrics,
  PostureMetrics,
} from '@/lib/golf/types';

interface MetricsPanelProps {
  analysis: SwingAnalysis;
  keyPositions: SwingAnalysis['keyPositions'];
  balance: BalanceMetrics;
  plane: PlaneMetrics;
  posture: PostureMetrics;
}

const MetricCard = React.memo(({
  title,
  value,
  unit,
  description,
  good,
}: {
  title: string;
  value: string | number;
  unit?: string;
  description?: string;
  good?: boolean;
}) => (
  <div className="bg-gray-50 rounded-lg p-4">
    <p className="text-sm text-gray-500 mb-1">{title}</p>
    <p className="text-2xl font-bold text-gray-900">
      {value}
      {unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}
    </p>
    {description && <p className="text-xs text-gray-500 mt-1">{description}</p>}
    {good !== undefined && (
      <div className="flex items-center mt-2">
        <div
          className={`w-2 h-2 rounded-full ${good ? 'bg-green-500' : 'bg-yellow-500'}`}
        />
        <span className="text-xs text-gray-500 ml-1">
          {good ? 'Good' : 'Needs attention'}
        </span>
      </div>
    )}
  </div>
));
MetricCard.displayName = 'MetricCard';

export default function MetricsPanel({
  analysis,
  keyPositions,
  balance,
  plane,
  posture,
}: MetricsPanelProps) {
  return (
    <div className="space-y-6">
      {/* Key Position Angles */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Key Position Angles
        </h3>

        {/* Address Position */}
        {keyPositions.address && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center">
              <span className="w-3 h-3 bg-slate-400 rounded-full mr-2" />
              Address Position
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <MetricCard
                title="Spine Angle"
                value={keyPositions.address.angles.spineAngle.toFixed(1)}
                unit="°"
                good={
                  keyPositions.address.angles.spineAngle >= 25 &&
                  keyPositions.address.angles.spineAngle <= 45
                }
              />
              <MetricCard
                title="Left Knee"
                value={keyPositions.address.angles.leftKneeFlexion.toFixed(1)}
                unit="°"
              />
              <MetricCard
                title="Right Knee"
                value={keyPositions.address.angles.rightKneeFlexion.toFixed(1)}
                unit="°"
              />
              <MetricCard
                title="Hip Rotation"
                value={keyPositions.address.angles.hipRotation.toFixed(1)}
                unit="°"
              />
            </div>
          </div>
        )}

        {/* Top of Backswing */}
        {keyPositions.top && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center">
              <span className="w-3 h-3 bg-indigo-500 rounded-full mr-2" />
              Top of Backswing
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <MetricCard
                title="Shoulder Turn"
                value={keyPositions.top.angles.shoulderRotation.toFixed(1)}
                unit="°"
                good={
                  keyPositions.top.angles.shoulderRotation >= 75 &&
                  keyPositions.top.angles.shoulderRotation <= 105
                }
              />
              <MetricCard
                title="Hip Turn"
                value={keyPositions.top.angles.hipRotation.toFixed(1)}
                unit="°"
                good={
                  keyPositions.top.angles.hipRotation >= 35 &&
                  keyPositions.top.angles.hipRotation <= 55
                }
              />
              <MetricCard
                title="X-Factor"
                value={keyPositions.top.angles.xFactor.toFixed(1)}
                unit="°"
                description="Shoulder-Hip differential"
                good={
                  keyPositions.top.angles.xFactor >= 40 &&
                  keyPositions.top.angles.xFactor <= 60
                }
              />
              <MetricCard
                title="Spine Angle"
                value={keyPositions.top.angles.spineAngle.toFixed(1)}
                unit="°"
              />
            </div>
          </div>
        )}

        {/* Impact Position */}
        {keyPositions.impact && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center">
              <span className="w-3 h-3 bg-red-500 rounded-full mr-2" />
              Impact Position
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <MetricCard
                title="Shoulder Rotation"
                value={keyPositions.impact.angles.shoulderRotation.toFixed(1)}
                unit="°"
              />
              <MetricCard
                title="Hip Rotation"
                value={keyPositions.impact.angles.hipRotation.toFixed(1)}
                unit="°"
                good={
                  keyPositions.impact.angles.hipRotation >= 30 &&
                  keyPositions.impact.angles.hipRotation <= 50
                }
              />
              <MetricCard
                title="X-Factor"
                value={keyPositions.impact.angles.xFactor.toFixed(1)}
                unit="°"
              />
              <MetricCard
                title="Left Elbow"
                value={keyPositions.impact.angles.leftElbowAngle.toFixed(1)}
                unit="°"
                good={keyPositions.impact.angles.leftElbowAngle >= 165}
              />
            </div>
          </div>
        )}

        {/* Finish Position */}
        {keyPositions.finish && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center">
              <span className="w-3 h-3 bg-emerald-500 rounded-full mr-2" />
              Finish Position
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <MetricCard
                title="Shoulder Rotation"
                value={keyPositions.finish.angles.shoulderRotation.toFixed(1)}
                unit="°"
              />
              <MetricCard
                title="Hip Rotation"
                value={keyPositions.finish.angles.hipRotation.toFixed(1)}
                unit="°"
              />
              <MetricCard
                title="Spine Angle"
                value={keyPositions.finish.angles.spineAngle.toFixed(1)}
                unit="°"
              />
              <MetricCard
                title="Hip Tilt"
                value={keyPositions.finish.angles.hipTilt.toFixed(1)}
                unit="°"
              />
            </div>
          </div>
        )}
      </div>

      {/* Balance Metrics */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Balance & Weight Shift</h3>

        {/* Weight Distribution Visualization */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-500 mb-2">Address</p>
            <div className="flex items-center space-x-2">
              <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500"
                  style={{ width: `${balance.addressWeight.left}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {balance.addressWeight.left.toFixed(0)}:{balance.addressWeight.right.toFixed(0)}
              </span>
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-500 mb-2">Top</p>
            <div className="flex items-center space-x-2">
              <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500"
                  style={{ width: `${balance.topWeight.left}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {balance.topWeight.left.toFixed(0)}:{balance.topWeight.right.toFixed(0)}
              </span>
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-500 mb-2">Impact</p>
            <div className="flex items-center space-x-2">
              <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-500"
                  style={{ width: `${balance.impactWeight.left}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {balance.impactWeight.left.toFixed(0)}:{balance.impactWeight.right.toFixed(0)}
              </span>
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-500 mb-2">Finish</p>
            <div className="flex items-center space-x-2">
              <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500"
                  style={{ width: `${balance.finishWeight.left}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {balance.finishWeight.left.toFixed(0)}:{balance.finishWeight.right.toFixed(0)}
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <MetricCard
            title="Sway Amount"
            value={balance.swayAmount.toFixed(1)}
            unit="cm"
            description="Lateral movement in backswing"
            good={balance.swayAmount < 10}
          />
          <MetricCard
            title="Slide Amount"
            value={balance.slideAmount.toFixed(1)}
            unit="cm"
            description="Lateral movement in downswing"
            good={balance.slideAmount < 15}
          />
          <MetricCard
            title="Hip Bump"
            value={balance.hipBump.toFixed(1)}
            unit="cm"
            description="Forward hip movement"
            good={balance.hipBump >= 5 && balance.hipBump <= 15}
          />
        </div>
      </div>

      {/* Plane Metrics */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Swing Plane Analysis</h3>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <MetricCard
            title="Backswing Plane"
            value={plane.backswingPlaneAngle.toFixed(1)}
            unit="°"
          />
          <MetricCard
            title="Downswing Plane"
            value={plane.downswingPlaneAngle.toFixed(1)}
            unit="°"
          />
          <MetricCard
            title="Plane Difference"
            value={plane.planeDifferential.toFixed(1)}
            unit="°"
            good={plane.planeDifferential < 10}
          />
          <MetricCard
            title="On Plane"
            value={plane.onPlane ? 'Yes' : 'No'}
            good={plane.onPlane}
          />
        </div>

        <div className="mt-4 grid grid-cols-3 gap-4">
          <MetricCard
            title="Shaft at Address"
            value={plane.shaftAngleAtAddress.toFixed(1)}
            unit="°"
          />
          <MetricCard
            title="Shaft at Top"
            value={plane.shaftAngleAtTop.toFixed(1)}
            unit="°"
          />
          <MetricCard
            title="Shaft at Impact"
            value={plane.shaftAngleAtImpact.toFixed(1)}
            unit="°"
          />
        </div>
      </div>

      {/* Posture Metrics */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Posture Analysis</h3>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <MetricCard
            title="Head Stability"
            value={posture.headStability.toFixed(0)}
            unit="%"
            good={posture.headStability >= 70}
          />
          <MetricCard
            title="Early Extension"
            value={posture.earlyExtension ? 'Detected' : 'None'}
            good={!posture.earlyExtension}
          />
          <MetricCard
            title="Loss of Posture"
            value={posture.lossOfPosture ? 'Detected' : 'None'}
            good={!posture.lossOfPosture}
          />
          <MetricCard
            title="Reverse Spine"
            value={posture.reverseSpineTilt ? 'Detected' : 'None'}
            good={!posture.reverseSpineTilt}
          />
        </div>

        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Address Posture</h4>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-gray-500">Spine Angle</p>
              <p className="text-lg font-medium text-gray-900">
                {posture.addressPosture.spineAngle.toFixed(1)}°
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Knee Flexion</p>
              <p className="text-lg font-medium text-gray-900">
                {posture.addressPosture.kneeFlexion.toFixed(1)}°
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Arm Hang</p>
              <p className="text-lg font-medium text-gray-900 capitalize">
                {posture.addressPosture.armHang.replace('_', ' ')}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
