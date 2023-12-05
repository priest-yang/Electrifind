-- MySQL dump 10.13  Distrib 8.0.35, for Linux (x86_64)
--
-- Host: 34.71.12.223    Database: NREL_all_stations_data
-- ------------------------------------------------------
-- Server version	8.0.27

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
SET @MYSQLDUMP_TEMP_LOG_BIN = @@SESSION.SQL_LOG_BIN;
SET @@SESSION.SQL_LOG_BIN= 0;

--
-- GTID state at the beginning of the backup 
--

SET @@GLOBAL.GTID_PURGED=/*!80000 '+'*/ '6737ac48-3f63-11ee-9725-eab9ff45b5c4:1-192430';

--
-- Table structure for table `cities`
--

DROP TABLE IF EXISTS `cities`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cities` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `cities`
--

LOCK TABLES `cities` WRITE;
/*!40000 ALTER TABLE `cities` DISABLE KEYS */;
INSERT INTO `cities` VALUES (1,'Ann Arbor');
/*!40000 ALTER TABLE `cities` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ev_connector_types`
--

DROP TABLE IF EXISTS `ev_connector_types`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ev_connector_types` (
  `id` int NOT NULL AUTO_INCREMENT,
  `type` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ev_connector_types`
--

LOCK TABLES `ev_connector_types` WRITE;
/*!40000 ALTER TABLE `ev_connector_types` DISABLE KEYS */;
INSERT INTO `ev_connector_types` VALUES (1,'J1772'),(2,'CHADEMO'),(3,'Tesla'),(4,'NEMA515');
/*!40000 ALTER TABLE `ev_connector_types` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ev_networks`
--

DROP TABLE IF EXISTS `ev_networks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ev_networks` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ev_networks`
--

LOCK TABLES `ev_networks` WRITE;
/*!40000 ALTER TABLE `ev_networks` DISABLE KEYS */;
INSERT INTO `ev_networks` VALUES (1,'Blink Network'),(2,'ChargePoint Network'),(3,'EV Connect'),(4,'FLO'),(5,'Non-Networked'),(6,'OpConnect'),(7,'SHELL_RECHARGE'),(8,'Tesla');
/*!40000 ALTER TABLE `ev_networks` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `facility_types`
--

DROP TABLE IF EXISTS `facility_types`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `facility_types` (
  `id` int NOT NULL AUTO_INCREMENT,
  `type` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `facility_types`
--

LOCK TABLES `facility_types` WRITE;
/*!40000 ALTER TABLE `facility_types` DISABLE KEYS */;
INSERT INTO `facility_types` VALUES (1,'CAR_DEALER'),(2,'FED_GOV'),(3,'HOTEL'),(4,'OFFICE_BLDG'),(5,'PARKING_GARAGE'),(6,'PARKING_LOT'),(7,'PAY_GARAGE');
/*!40000 ALTER TABLE `facility_types` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `states`
--

DROP TABLE IF EXISTS `states`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `states` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(2) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `states`
--

LOCK TABLES `states` WRITE;
/*!40000 ALTER TABLE `states` DISABLE KEYS */;
INSERT INTO `states` VALUES (1,'MI');
/*!40000 ALTER TABLE `states` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `stations`
--

DROP TABLE IF EXISTS `stations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `stations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` tinytext,
  `street_address` tinytext,
  `city` int DEFAULT NULL,
  `state` int DEFAULT NULL,
  `zip` int DEFAULT NULL,
  `ev_level1_evse_num` int DEFAULT NULL,
  `ev_level2_evse_num` int DEFAULT NULL,
  `ev_dc_fast_count` int DEFAULT NULL,
  `ev_network` int DEFAULT NULL,
  `date_last_confirmed` datetime DEFAULT NULL,
  `facility_type` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `stations_cities_id_fk` (`city`),
  KEY `stations_states_id_fk` (`state`),
  KEY `stations_ev_networks_id_fk` (`ev_network`),
  KEY `stations_facility_types_id_fk` (`facility_type`),
  KEY `stations_zipcodes_id_fk` (`zip`),
  CONSTRAINT `stations_cities_id_fk` FOREIGN KEY (`city`) REFERENCES `cities` (`id`),
  CONSTRAINT `stations_ev_networks_id_fk` FOREIGN KEY (`ev_network`) REFERENCES `ev_networks` (`id`),
  CONSTRAINT `stations_facility_types_id_fk` FOREIGN KEY (`facility_type`) REFERENCES `facility_types` (`id`),
  CONSTRAINT `stations_states_id_fk` FOREIGN KEY (`state`) REFERENCES `states` (`id`),
  CONSTRAINT `stations_zipcodes_id_fk` FOREIGN KEY (`zip`) REFERENCES `zipcodes` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=77 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `stations`
--

LOCK TABLES `stations` WRITE;
/*!40000 ALTER TABLE `stations` DISABLE KEYS */;
INSERT INTO `stations` VALUES (1,'Ann Arbor Downtown Development Authority - Catherine and Fourth Surface Lot','121 Catherine St',1,1,2,NULL,2,NULL,5,'2023-06-12 00:00:00',6),(2,'Ann Arbor Nissan','3975 Jackson Rd',1,1,1,NULL,1,2,5,'2022-03-07 00:00:00',1),(3,'Ann Arbor Nissan','3975 Jackson Rd',1,1,1,NULL,1,NULL,5,'2022-03-07 00:00:00',1),(4,'IMRA America','1044 Woodridge Ave',1,1,3,NULL,2,NULL,5,'2022-06-14 00:00:00',4),(5,'Varsity Ford','3480 Jackson Rd',1,1,1,NULL,1,NULL,5,'2022-06-14 00:00:00',1),(6,'Ann Arbor Downtown Development Authority - Ashley and Washington Parking Structure','215 W Washington',1,1,2,NULL,3,NULL,5,'2022-05-05 00:00:00',7),(7,'First Martin','115 Depot St',1,1,2,NULL,1,NULL,5,'2023-07-11 00:00:00',4),(8,'Meijer - Tesla Supercharger','3145 Ann Arbor-Saline Road',1,1,1,NULL,NULL,8,8,'2023-07-01 00:00:00',NULL),(9,'Sheraton Ann Arbor Hotel - Tesla Destination','3200 Boardwalk Dr',1,1,4,NULL,4,NULL,8,'2022-10-06 00:00:00',3),(10,'173 - Ann Arbor','5645 Jackson Road',1,1,1,NULL,4,2,7,'2023-11-14 00:00:00',NULL),(11,'MEADOWLARK BLDG STATION 1','3250 W Liberty Rd',1,1,1,NULL,1,NULL,2,'2023-11-14 00:00:00',NULL),(12,'EPA Ann Arbor - Station 1','2565 Plymouth Rd',1,1,3,NULL,6,NULL,5,'2021-02-22 00:00:00',2),(13,'EPA Ann Arbor - Station 2','2565 Plymouth Rd',1,1,3,1,NULL,NULL,5,'2021-02-22 00:00:00',2),(14,'BMW ANN ARBOR STATION 01','501 Auto Mall Dr',1,1,1,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(15,'Prentice Partners','830 Henry Street',1,1,2,NULL,10,NULL,7,'2023-11-14 00:00:00',NULL),(16,'Hoover and Greene','950 Greene St',1,1,2,NULL,2,NULL,3,'2023-11-14 00:00:00',NULL),(17,'BEEKMAN STATION 1','1160 Broadway St',1,1,3,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(18,'WASHTENAW BP 1','4975 Washtenaw Ave',1,1,4,NULL,NULL,1,2,'2023-11-14 00:00:00',NULL),(19,'Suburban Chevrolet','3515 Jackson Rd',1,1,1,NULL,2,NULL,5,'2022-05-05 00:00:00',1),(20,'Audi Ann Arbor','2575 S State St',1,1,2,NULL,2,NULL,5,'2022-05-05 00:00:00',1),(21,'ProQuest Employee Parking Garage','789 E Eisenhower Pkwy',1,1,4,NULL,4,NULL,5,'2022-05-05 00:00:00',5),(22,'Staybridge Suites','3850 Research Park Dr',1,1,4,NULL,1,NULL,5,'2022-05-05 00:00:00',3),(23,'Mitsubishi Motor - Ann Arbor Lab','3735 Varsity Dr',1,1,4,NULL,2,NULL,5,'2022-05-05 00:00:00',6),(24,'A2DDA STATION 2','324 Maynard St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(25,'A2DDA STATION 3','324 Maynard St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(26,'A2DDA STATION 1','324 Maynard St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(27,'A2DDA STATION 4','324 Maynard St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(28,'A2DDA ST 4123','320 Thompson St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(29,'A2DDA STATION 4121','320 Thompson St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(30,'A2DDA 500 E WASH 1','500 E Washington St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(31,'A2DDA 500 E WASH 2','500 E Washington St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(32,'A2DDA STATION 27','123E W William St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(33,'A2DDA STATION 28','115E W William St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(34,'A2DDA STATION 22','115E W William St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(35,'A2DDA STATION 24','115E W William St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(36,'A2DDA STATION 18','220 N Ashley St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(37,'A2DDA STATION 19','220 N Ashley St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(38,'A2DDA STATION 20','220 N Ashley St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(39,'A2DDA STATION 13','120 W Ann St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(40,'A2DDA STATION 17','220 N Ashely St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(41,'A2DDA STATION 15','220 N Ashley St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(42,'A2DDA STATION 12','120 W Ann St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(43,'A2DDA STATION 26','650 S Forest Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(44,'A2DDA STATION 8','650 S Forest Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(45,'A2DDA STATION 21','650 S Forest Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(46,'A2DDA STATION 16','650 S Forest Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(47,'A2DDA STATION 25','650 S Forest Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(48,'A2DDA STATION 23','650 S Forest Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(49,'A2DDA STATION 6','650 S Forest Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(50,'A2DDA STATION 31','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(51,'A2DDA STATION 11','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(52,'A2DDA STATION 9','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(53,'A2DDA STATION 29','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(54,'A2DDA STATION 32','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(55,'A2DDA STATION 30','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(56,'A2DDA STATION 7','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(57,'A2DDA STATION 5','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(58,'A2DDA STATION 10','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(59,'A2DDA 5TH AVE CT4K','319 S 5th Ave',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(60,'A2DDA E WASH CT4K','123 E Washington St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(61,'A2DDA E WASH CT4K 2','123 E Washington St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(62,'A2DDA STATION 14','220 N Ashley St',1,1,2,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(63,'OWL CREEK FAST CHARGER','3400 Nixon Rd',1,1,3,NULL,NULL,1,2,'2023-11-14 00:00:00',NULL),(64,'WEBER\'S WEBER\'S HOTEL 1','3050 Jackson Ave',1,1,1,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(65,'HYAA STATION 1','4001 Jackson Rd',1,1,1,NULL,NULL,1,2,'2023-11-14 00:00:00',NULL),(66,'FMC-MCT SOUTHEAST','3120 Chelsea Cir',1,1,4,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(67,'FMC-MCT SOUTHWEST','3126 Chelsea Cir',1,1,4,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(68,'FMC-MCT NORTHEAST','3120 Chelsea Cir',1,1,4,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(69,'FMC-MCT NORTHWEST','3120 Chelsea Cir',1,1,4,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(70,'WASHCOMMCOLLEGE PARKING ST 2','4800 E Huron river dr',1,1,3,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(71,'Baker Commons-Forth-Public','106 Packard St',1,1,5,NULL,1,NULL,6,'2023-11-14 00:00:00',NULL),(72,'Baker Commons-Forth-Carshare','106 Packard St',1,1,5,NULL,1,NULL,6,'2023-11-14 00:00:00',NULL),(73,'FA ANN ARBOR FOX ACURA #2','540 Auto Mall Dr',1,1,1,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(74,'FA ANN ARBOR FOX ACURA #1','540 Auto Mall Dr',1,1,1,NULL,2,NULL,2,'2023-11-14 00:00:00',NULL),(75,'Farah Professional Center','3100 West Liberty Road',1,1,1,NULL,2,NULL,1,'2023-11-14 00:00:00',NULL),(76,'407 North Ingalls','407 N Ingalls St',1,1,2,NULL,1,NULL,4,'2023-11-14 00:00:00',NULL);
/*!40000 ALTER TABLE `stations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `stations_connectors`
--

DROP TABLE IF EXISTS `stations_connectors`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `stations_connectors` (
  `id` int NOT NULL AUTO_INCREMENT,
  `station_id` int DEFAULT NULL,
  `connector_type` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `stations_connectors_ev_connector_types_id_fk` (`connector_type`),
  KEY `stations_connectors_stations_id_fk` (`station_id`),
  CONSTRAINT `stations_connectors_ev_connector_types_id_fk` FOREIGN KEY (`connector_type`) REFERENCES `ev_connector_types` (`id`),
  CONSTRAINT `stations_connectors_stations_id_fk` FOREIGN KEY (`station_id`) REFERENCES `stations` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=80 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `stations_connectors`
--

LOCK TABLES `stations_connectors` WRITE;
/*!40000 ALTER TABLE `stations_connectors` DISABLE KEYS */;
INSERT INTO `stations_connectors` VALUES (1,1,1),(2,2,2),(3,2,1),(4,3,1),(5,4,1),(6,5,1),(7,6,1),(8,7,1),(9,8,3),(10,9,1),(11,9,3),(12,10,2),(13,10,1),(14,11,1),(15,12,1),(16,13,4),(17,14,1),(18,15,1),(19,16,1),(20,17,1),(21,18,2),(22,19,1),(23,20,1),(24,21,1),(25,22,1),(26,23,1),(27,24,1),(28,25,1),(29,26,1),(30,27,1),(31,28,1),(32,29,1),(33,30,1),(34,31,1),(35,32,1),(36,33,1),(37,34,1),(38,35,1),(39,36,1),(40,37,1),(41,38,1),(42,39,1),(43,40,1),(44,41,1),(45,42,1),(46,43,1),(47,44,1),(48,45,1),(49,46,1),(50,47,1),(51,48,1),(52,49,1),(53,50,1),(54,51,1),(55,52,1),(56,53,1),(57,54,1),(58,55,1),(59,56,1),(60,57,1),(61,58,1),(62,59,1),(63,60,1),(64,61,1),(65,62,1),(66,63,2),(67,64,1),(68,65,2),(69,66,1),(70,67,1),(71,68,1),(72,69,1),(73,70,1),(74,71,1),(75,72,1),(76,73,1),(77,74,1),(78,75,1),(79,76,1);
/*!40000 ALTER TABLE `stations_connectors` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `zipcodes`
--

DROP TABLE IF EXISTS `zipcodes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `zipcodes` (
  `id` int NOT NULL AUTO_INCREMENT,
  `code` varchar(5) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `zipcodes`
--

LOCK TABLES `zipcodes` WRITE;
/*!40000 ALTER TABLE `zipcodes` DISABLE KEYS */;
INSERT INTO `zipcodes` VALUES (1,'48103'),(2,'48104'),(3,'48105'),(4,'48108'),(5,'48509');
/*!40000 ALTER TABLE `zipcodes` ENABLE KEYS */;
UNLOCK TABLES;
SET @@SESSION.SQL_LOG_BIN = @MYSQLDUMP_TEMP_LOG_BIN;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-12-02  1:56:27
