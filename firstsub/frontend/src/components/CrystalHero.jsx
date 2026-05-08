import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function CrystalHero() {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 0.35, 7);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const crystal = new THREE.Group();
    const material = new THREE.MeshPhysicalMaterial({
      color: 0x7be8ff,
      emissive: 0x06354a,
      metalness: 0.04,
      roughness: 0.18,
      transmission: 0.35,
      thickness: 0.8,
      clearcoat: 0.9,
      clearcoatRoughness: 0.2,
    });
    const goldMaterial = new THREE.MeshStandardMaterial({ color: 0xffc675, metalness: 0.45, roughness: 0.32 });

    const core = new THREE.Mesh(new THREE.OctahedronGeometry(1.8, 1), material);
    core.scale.set(1, 1.35, 0.82);
    crystal.add(core);

    const facetGeometry = new THREE.TetrahedronGeometry(0.55, 0);
    const anchors = [
      [-1.55, -0.5, 0.25],
      [1.52, -0.38, -0.1],
      [-0.52, 1.68, -0.22],
      [0.72, 1.42, 0.38],
      [0.22, -1.78, 0.1],
    ];
    anchors.forEach(([x, y, z], index) => {
      const shard = new THREE.Mesh(facetGeometry, index % 2 ? material : goldMaterial);
      shard.position.set(x, y, z);
      shard.rotation.set(index * 0.72, index * 0.41, index * 0.28);
      crystal.add(shard);
    });

    scene.add(crystal);
    scene.add(new THREE.AmbientLight(0xbddfff, 1.8));
    const cyan = new THREE.PointLight(0x5ce5ff, 22, 16);
    cyan.position.set(-3, 3, 5);
    scene.add(cyan);
    const gold = new THREE.PointLight(0xffc675, 12, 12);
    gold.position.set(3, -1.5, 4);
    scene.add(gold);

    const pointer = { x: 0, y: 0 };
    const resize = () => {
      const { clientWidth, clientHeight } = mount;
      renderer.setSize(clientWidth, clientHeight, false);
      camera.aspect = clientWidth / Math.max(clientHeight, 1);
      camera.updateProjectionMatrix();
    };
    const onPointerMove = (event) => {
      const rect = mount.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width - 0.5) * 0.55;
      pointer.y = ((event.clientY - rect.top) / rect.height - 0.5) * 0.45;
    };

    resize();
    window.addEventListener("resize", resize);
    mount.addEventListener("pointermove", onPointerMove);

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      crystal.rotation.y += 0.006;
      crystal.rotation.x += (pointer.y - crystal.rotation.x) * 0.035;
      crystal.rotation.z += (pointer.x - crystal.rotation.z) * 0.025;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      mount.removeEventListener("pointermove", onPointerMove);
      renderer.dispose();
      material.dispose();
      goldMaterial.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  return <div className="crystal-hero" ref={mountRef} aria-hidden="true" />;
}
