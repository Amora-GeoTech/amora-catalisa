/*==============================================================================

  Copyright (c) Kitware, Inc.

  See http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Julien Finet, Kitware, Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// AmoraDigitalRock includes
#include "qAmoraDigitalRockAppMainWindow.h"
#include "qAmoraDigitalRockAppMainWindow_p.h"

// Qt includes
#include <QDesktopWidget>
#include <QLabel>
#include <QToolBar>

// Slicer includes
#include "qSlicerApplication.h"
#include "qSlicerAboutDialog.h"
#include "qSlicerMainWindow_p.h"
#include "qSlicerModuleSelectorToolBar.h"
#include "qMRMLWidget.h"

//-----------------------------------------------------------------------------
// qAmoraDigitalRockAppMainWindowPrivate methods

qAmoraDigitalRockAppMainWindowPrivate::qAmoraDigitalRockAppMainWindowPrivate(qAmoraDigitalRockAppMainWindow& object)
  : Superclass(object)
{
}

//-----------------------------------------------------------------------------
qAmoraDigitalRockAppMainWindowPrivate::~qAmoraDigitalRockAppMainWindowPrivate()
{
}

//-----------------------------------------------------------------------------
void qAmoraDigitalRockAppMainWindowPrivate::init()
{
#if (QT_VERSION >= QT_VERSION_CHECK(5, 7, 0))
  QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
  Q_Q(qAmoraDigitalRockAppMainWindow);
  this->Superclass::init();
}

//-----------------------------------------------------------------------------
void qAmoraDigitalRockAppMainWindowPrivate::setupUi(QMainWindow * mainWindow)
{
  qSlicerApplication * app = qSlicerApplication::application();

  //----------------------------------------------------------------------------
  // Add actions
  //----------------------------------------------------------------------------
  QAction* helpAboutSlicerAppAction = new QAction(mainWindow);
  helpAboutSlicerAppAction->setObjectName("HelpAboutAmoraDigitalRockAppAction");
  helpAboutSlicerAppAction->setText(qAmoraDigitalRockAppMainWindow::tr("About %1").arg(qSlicerApplication::application()->mainApplicationDisplayName()));

  //----------------------------------------------------------------------------
  // Calling "setupUi()" after adding the actions above allows the call
  // to "QMetaObject::connectSlotsByName()" done in "setupUi()" to
  // successfully connect each slot with its corresponding action.
  this->Superclass::setupUi(mainWindow);

  // Add Help Menu Action
  this->HelpMenu->addAction(helpAboutSlicerAppAction);

  //----------------------------------------------------------------------------
  // Configure
  //----------------------------------------------------------------------------
  mainWindow->setWindowIcon(QIcon(":/Icons/Medium/DesktopIcon.png"));

  QLabel* logoLabel = new QLabel();
  logoLabel->setObjectName("LogoLabel");
  logoLabel->setPixmap(qMRMLWidget::pixmapFromIcon(QIcon(":/Icons/Medium/DesktopIcon.png")));
  this->PanelDockWidget->setTitleBarWidget(logoLabel);

  // Keep the menu bar visible — AMORA toolbar supplements but doesn't replace it

  // Hide the old-style module toolbar icons (ModuleToolBar)
  QToolBar* moduleToolBar = mainWindow->findChild<QToolBar*>("ModuleToolBar");
  if (moduleToolBar) { moduleToolBar->setVisible(false); }

  // Keep the ModuleSelectorToolBar visible - this is the "Modules:" dropdown
  // that allows switching between AMORA modules (Digital Rock, Filtering, Processing)
  // QToolBar* moduleSelectorToolBar = mainWindow->findChild<QToolBar*>("ModuleSelectorToolBar");
  // if (moduleSelectorToolBar) { moduleSelectorToolBar->setVisible(false); }
}

//-----------------------------------------------------------------------------
// qAmoraDigitalRockAppMainWindow methods

//-----------------------------------------------------------------------------
qAmoraDigitalRockAppMainWindow::qAmoraDigitalRockAppMainWindow(QWidget* windowParent)
  : Superclass(new qAmoraDigitalRockAppMainWindowPrivate(*this), windowParent)
{
  Q_D(qAmoraDigitalRockAppMainWindow);
  d->init();
}

//-----------------------------------------------------------------------------
qAmoraDigitalRockAppMainWindow::qAmoraDigitalRockAppMainWindow(
  qAmoraDigitalRockAppMainWindowPrivate* pimpl, QWidget* windowParent)
  : Superclass(pimpl, windowParent)
{
  // init() is called by derived class.
}

//-----------------------------------------------------------------------------
qAmoraDigitalRockAppMainWindow::~qAmoraDigitalRockAppMainWindow()
{
}

//-----------------------------------------------------------------------------
void qAmoraDigitalRockAppMainWindow::on_HelpAboutAmoraDigitalRockAppAction_triggered()
{
  qSlicerAboutDialog about(this);
  about.setLogo(QPixmap(":/Logo.png"));
  about.exec();
}
