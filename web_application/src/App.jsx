import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Hero from "./components/hero"
import Navbar from "./components/Navbar"
import AOS from 'aos';
import 'aos/dist/aos.css';
import React, { useEffect } from "react";
import Recherche from "./components/Recherche";
import Modele from "./components/Modele";
import Demo from "./components/Demo";
import Pipeline from "./pages/pipeline";



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
        {/* Page principale */}
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

        {/* Page Pipeline */}
        <Route path="/pipeline" element={<Pipeline />} />
      </Routes>

    </Router>
  )
}

export default App