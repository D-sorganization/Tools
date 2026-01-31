import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

const URDFViewer = () => {
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [joints, setJoints] = useState({});
    const [showCollision, setShowCollision] = useState(false);

    const viewerRef = useRef(null);
    const sceneRef = useRef(null); // To store scene, camera, renderer, etc.
    const robotRef = useRef(null);

    // Fetch models on mount
    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            setModels(data.models);
        } catch (error) {
            console.error('Failed to fetch models:', error);
        }
    };

    const handleUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        setLoading(true);
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                await fetchModels();
                setSelectedModel(data.filename);
            } else {
                console.error('Upload failed');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleJointChange = (name, value) => {
        const val = parseFloat(value);
        setJoints(prev => ({
            ...prev,
            [name]: { ...prev[name], value: val }
        }));

        if (robotRef.current && robotRef.current.joints[name]) {
            robotRef.current.joints[name].setJointValue(val);
        }
    };

    // Toggle collision visibility
    useEffect(() => {
        if (robotRef.current) {
            robotRef.current.traverse(child => {
                // urdf-loader typically marks collision groups with isURDFCollision = true
                if (child.isURDFCollision) {
                    child.visible = showCollision;
                }
            });
        }
    }, [showCollision]);

    // Initialize Three.js scene
    useEffect(() => {
        if (!viewerRef.current || !selectedModel) return;

        // Cleanup previous scene if exists
        if (sceneRef.current) {
            if (viewerRef.current.innerHTML) {
                viewerRef.current.innerHTML = '';
            }
            sceneRef.current.renderer.dispose();
            setJoints({});
            robotRef.current = null;
        }

        const width = viewerRef.current.clientWidth;
        const height = viewerRef.current.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x2a2a2a);

        // Grid
        const gridHelper = new THREE.GridHelper(10, 10);
        scene.add(gridHelper);

        // Axes
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);

        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        camera.position.set(2, 2, 2);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        viewerRef.current.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5);
        scene.add(directionalLight);

        // Load URDF
        const manager = new THREE.LoadingManager();
        const loader = new URDFLoader(manager);

        loader.load(
            `/api/models/${selectedModel}`,
            (robot) => {
                robotRef.current = robot;
                scene.add(robot);
                robot.rotation.x = -Math.PI / 2;

                // Extract joints
                const jointList = {};
                for (const name in robot.joints) {
                    const joint = robot.joints[name];
                    if (joint._jointType !== 'fixed') {
                        let min = joint.limit.lower;
                        let max = joint.limit.upper;
                        if (joint._jointType === 'continuous') {
                            min = -Math.PI;
                            max = Math.PI;
                        }

                        jointList[name] = {
                            value: joint.angle || 0,
                            min: min,
                            max: max,
                            type: joint._jointType
                        };
                    }
                }
                setJoints(jointList);

                // Initial collision state
                robot.traverse(child => {
                    if (child.isURDFCollision) {
                        child.visible = showCollision;
                    }
                });

                // Try to frame the robot
                const box = new THREE.Box3().setFromObject(robot);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / 2 * Math.tan(fov * 2));
                cameraZ = Math.max(cameraZ, 2.0);
                cameraZ *= 2.0;

                camera.position.set(center.x + cameraZ, center.y + cameraZ, center.z + cameraZ);
                camera.lookAt(center);
                controls.target.copy(center);
                controls.update();

                console.log('Robot loaded:', robot);
            },
            undefined,
            (error) => {
                console.error('An error happened', error);
            }
        );

        let animationId;
        const animate = () => {
            animationId = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        sceneRef.current = { scene, camera, renderer, controls };

        const handleResize = () => {
            if (!viewerRef.current) return;
            const w = viewerRef.current.clientWidth;
            const h = viewerRef.current.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            cancelAnimationFrame(animationId);
            if (viewerRef.current) {
                viewerRef.current.innerHTML = '';
            }
            renderer.dispose();
            robotRef.current = null;
        };

    }, [selectedModel]);

    return (
        <div className="app-container">
            <div className="sidebar">
                <h2>URDF Viewer</h2>
                <div className="upload-section">
                    <label className="upload-btn">
                        Upload URDF
                        <input type="file" accept=".urdf,.xml" onChange={handleUpload} style={{display: 'none'}} />
                    </label>
                </div>

                <div className="models-list">
                    <h3>Models</h3>
                    {loading && <p>Loading...</p>}
                    <ul>
                        {models.map(model => (
                            <li
                                key={model}
                                className={selectedModel === model ? 'active' : ''}
                                onClick={() => setSelectedModel(model)}
                            >
                                {model}
                            </li>
                        ))}
                    </ul>
                </div>

                {Object.keys(joints).length > 0 && (
                    <div className="controls-section">
                        <h3>Joints</h3>
                        <div className="control-item">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={showCollision}
                                    onChange={(e) => setShowCollision(e.target.checked)}
                                />
                                Show Collision
                            </label>
                        </div>
                        {Object.keys(joints).map(name => (
                            <div key={name} className="control-item">
                                <label style={{display: 'block', marginBottom: '5px', fontSize: '0.9em'}}>{name}</label>
                                <input
                                    type="range"
                                    min={joints[name].min}
                                    max={joints[name].max}
                                    step="0.01"
                                    value={joints[name].value}
                                    onChange={(e) => handleJointChange(name, e.target.value)}
                                    style={{width: '100%'}}
                                />
                                <div style={{textAlign: 'right', fontSize: '0.8em', color: '#aaa'}}>
                                    {joints[name].value.toFixed(2)} rad
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
            <div className="viewer-container" ref={viewerRef}>
                {!selectedModel && <div className="placeholder">Select or upload a model to view</div>}
            </div>
        </div>
    );
};

const root = createRoot(document.getElementById('root'));
root.render(<URDFViewer />);
