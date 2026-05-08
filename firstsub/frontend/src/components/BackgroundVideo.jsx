// BackgroundVideo.jsx
import React from 'react';

export default function BackgroundVideo() {
  return (
    <>
      <video
        autoPlay
        loop
        muted
        playsInline
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          zIndex: 0,
          filter: "blur(6px)",
          opacity: 0.85,
        }}
      >
        <source src="/background.mp4" type="video/mp4" />
      </video>
      <div 
        style={{
          position: "absolute",
          inset: 0,
          backgroundColor: "rgba(240, 242, 248, 0.3)",
          zIndex: 0,
          pointerEvents: "none"
        }}
      />
    </>
  );
}
