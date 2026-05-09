import { Menu, X, Github } from "lucide-react";
import { useState } from "react";

const navItems = [
  { title: 'Accueil', href: '#' },
  { title: 'Recherche', href: '#Recherche' },
  { title: 'Modèle', href: '#Modele' },
  { title: 'Démonstration', href: '#Demo' },
  { title: 'Résultats', href: '#Resultats' },
  { title: 'À propos', href: '#About' },
]

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  
  return (
    <nav className="fixed w-full bg-white/95 backdrop-blur-sm top-0 left-0 right-0 z-50 border-b border-gray-100 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-12 lg:px-20 py-3 
        flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center">
                <img
                    src="/src/assets/logo2.png"
                    alt="Logo du site"
                    className="w-10 sm:w-12 h-auto object-contain cursor-pointer
                    transition-all duration-500 ease-in-out hover:scale-110 hover:drop-shadow-md"
                />
            </div>

            {/* Navigation Desktop */}
            <ul className="hidden md:flex items-center gap-6 lg:gap-8 
            text-gray-700 font-medium">
                {navItems.map(({title, href}) => (
                    <li key={title}>
                        <a 
                          href={href} 
                          className="hover:text-blue-600 cursor-pointer 
                          transition-colors relative group"
                        >
                            {title}
                            <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-blue-600 
                            transition-all duration-300 group-hover:w-full"></span>
                        </a>
                    </li>
                ))}
            </ul>

            {/* Bouton CTA Desktop */}
            <div className="hidden md:flex items-center gap-3">
                <a 
                  href="https://github.com/Sxsthxnx17/Projet-TER-F-One-shot-learning" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="px-4 py-2 rounded-lg border border-gray-300
                  text-gray-700 font-medium hover:bg-gray-50
                  transition-colors flex items-center gap-2"
                >
                  <Github className="w-4 h-4" />
                  <span className="hidden lg:inline">Code</span>
                </a>
                <button 
                  onClick={() => window.location.href = '#Demo'}
                  className="px-4 py-2 sm:py-2 rounded-lg sm:rounded-xl
                  bg-blue-600 text-white font-medium hover:bg-blue-700 
                  transition-colors shadow-sm hover:shadow-md"
                >
                  Essayer la Démo
                </button>
            </div>

            {/* Menu Mobile Button */}
            <div className="md:hidden">
                <button  
                onClick={toggleMenu}
                className="p-1 rounded-md focus:outline-none 
                focus:ring-2 focus:ring-inset focus:ring-blue-600">
                    {isMenuOpen ? (
                        <X className="h-6 w-6 text-gray-700" />
                    ) : (
                        <Menu className="h-6 w-6 text-gray-700" />
                    )}                       
                </button>
            </div>
        </div>

        {/* Menu Mobile */}
        {isMenuOpen && (
            <div className="md:hidden bg-white shadow-lg border-t 
            border-gray-200">
                <div className="px-4 py-3 space-y-3">
                    {navItems.map(({title, href}) => (
                        <a 
                        key={title} 
                        href={href}
                        className="block py-2 px-4 text-gray-700 
                        hover:bg-blue-50 rounded-lg hover:text-blue-600 
                        transition-colors"
                        onClick={() => setIsMenuOpen(false)}> 
                            {title} 
                        </a>
                    ))}
                    <div className="pt-2 space-y-2">
                        <a 
                          href="https://github.com/Sxsthxnx17/Projet-TER-F-One-shot-learning" 
                          target="_blank"
                          rel="noopener noreferrer"
                          className="w-full py-2 rounded-lg border border-gray-300
                          text-gray-700 font-medium hover:bg-gray-50
                          transition-colors flex items-center justify-center gap-2"
                        >
                          <Github className="w-4 h-4" />
                          Code Source
                        </a>
                        <button  
                          onClick={() => {
                            window.location.href = '#Demo';
                            setIsMenuOpen(false);
                          }}
                          className="w-full py-2 rounded-lg
                          bg-blue-600 text-white font-medium hover:bg-blue-700 
                          transition-colors shadow-sm"
                        >
                            Essayer la Démo
                        </button>
                    </div>
                </div>
            </div>
        )} 
    </nav>
  )
}

export default Navbar