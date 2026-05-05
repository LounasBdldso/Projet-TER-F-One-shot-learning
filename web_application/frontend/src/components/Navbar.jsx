import { Menu,X } from "lucide-react";
import { useState } from "react";

const navItems = [
  { tittle: 'Accueil', href: '#' },
  { tittle: 'Recherche', href: '#Recherche' },
  { tittle: 'Modele', href: '#Modele' },
  { tittle: 'Démonstration', href: '#Demo' },
  { tittle: 'Contact', href: '#Contact' },
]

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  return (
    <nav className="fixed w-full bg-gray-50 top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-12 lg:px-20 py-3 
        flex items-center justify-between">
            {/* logo */}
            {/*<div className="flex items-center text-xl sm:text-2xl 
            font-bold text-gray-900 ">
                
                <div className="w-8 h-8 sm:h-10 sm:w-10 flex items-center justify-center 
                rounded-full border-3 sm:border-4 border-blue-950
                text-blue-950 mr-2">  
                    KI    
                </div>
                <span className="">
                    est<span className="text-yellow-500">
                        qui
                    </span>

                </span>
            </div>*/}
            <div className="flex items-center">
                <img
                    src="/src/assets/logo2.png"
                    alt="Logo du site"
                    className="w-10 sm:w-15 h-auto object-contain cursor-pointer
                    transition-all duration-500 ease-in-out hover:scale-105 hover:drop-shadow-md"
                />
            </div>

                {/* href */}
                <ul className="hidden md:flex  items-center gap-6 lg:gap-8 
                text-gray-700 font-medium">
                    {navItems.map(({tittle,href}) => (
                        <li key={tittle} >
                            <a href={href} className="hover:text-blue-800 cursor-pointer 
                            transition-colors ">
                                {tittle}
                            </a>
                        </li>
                    ))}

                </ul>
                {/* Boutons */} 
                <div className="hidden md:block">
                    <button  className=" px-4 py-2 sm:py-2 rounded-lg sm:rounded-xl
                    bg-blue-500 text-white font-medium hover:bg-blue-800 
                    transition-colors ">
                        Allons-y
                    </button>
                </div>

                {/* Menu */} 
                <div className="md:hidden">
                    <button  
                    onClick ={toggleMenu}
                    className=" p-1 rounded-md focus:outline-none 
                    focus:ring-2 focus:ring-inset focus:ring-blue-950 ">
                        {isMenuOpen ? (
                            < X className="h-6 w-6 text-gray-700"  />
                        ):(
                            < Menu className="h-6 w-6 text-gray-700"  />)}                       
                    </button>
                </div>
        </div>
        {isMenuOpen && (
            <div className="md:hidden bg-white shadow-lg border-t 
            border-gray-200">
                <div className="px-4 py-3 space-y-3">
                    {navItems.map(({tittle,href}) => (
                        <a 
                        key={tittle} 
                        href={href}
                        className="block py-2 px-4 text-gray-700 
                        hover:bg-gray-100 rounded-lg hover:text-blue-800 
                        transition-colors"
                        onClick={()=> setIsMenuOpen(false)}> 
                            {tittle} 
                        </a>

                    ))}
                    <div className="pt-2">
                        <button  className="w-full  py-2 rounded-lg
                        bg-blue-700 text-white font-medium hover:bg-blue-800 
                        transition-colors ">
                            Allons-y
                        </button>

                    </div>
                </div>
            </div>
        

                
            
       )} 
    </nav>
  )
}

export default Navbar