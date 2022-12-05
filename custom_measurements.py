import math

def measure_circularity(obj_area: float, obj_perimeter: float) -> float:
    """Return circularity of an object based on its area and perimeter. 
    A perfect circle as a circularity of 1.0.
    
    Args:
        obj_area (float): Area of the object.
        obj_perimeter (float): Perimeter of the object.
    
    Returns:
        float: Circularity of the object.
    """
    return 4*math.pi*(obj_area/obj_perimeter**2)