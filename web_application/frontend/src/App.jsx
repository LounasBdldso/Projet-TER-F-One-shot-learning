import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import React, { useEffect } from "react";
import AOS from 'aos';
import 'aos/dist/aos.css';

import Navbar from "./components/Navbar";
import Hero from "./components/hero"; // Remis en minuscule comme ton code d'origine
import Recherche from "./components/Recherche";
import Modele from "./components/Modele";
import Demo from "./components/Demo";

import Pipeline from "./pages/pipeline"; // Remis en minuscule
import DemoReconnaissance from "./pages/DemoReconnaissance";
import DemoQualite from "./pages/DemoQualite";

const App = () => {
  useEffect(() => {
    AOS.init({
      duration: 1000,
      easing: 'ease-in-out',
      once: true,
      offset: 100,
    });
  }, []);

  return (
    <Router>
      <Navbar />

      <Routes>
        <Route
          path="/"
          element={
            <>
              <Hero />
              <Recherche />
              <Modele />
              <Demo />
            </>
          }
        />

        <Route path="/pipeline" element={<Pipeline />} />
        
        {/* Assure-toi que ces deux fichiers existent avec CES majuscules */}
        <Route path="/demo/reconnaissance" element={<DemoReconnaissance />} />
        <Route path="/demo/qualite" element={<DemoQualite />} />
      </Routes>
    </Router>
  );
}

export default App;