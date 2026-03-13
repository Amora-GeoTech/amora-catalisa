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

#ifndef __qAmoraDigitalRockAppMainWindow_h
#define __qAmoraDigitalRockAppMainWindow_h

// AmoraDigitalRock includes
#include "qAmoraDigitalRockAppExport.h"
class qAmoraDigitalRockAppMainWindowPrivate;

// Slicer includes
#include "qSlicerMainWindow.h"

class Q_AMORADIGITALROCK_APP_EXPORT qAmoraDigitalRockAppMainWindow : public qSlicerMainWindow
{
  Q_OBJECT
public:
  typedef qSlicerMainWindow Superclass;

  qAmoraDigitalRockAppMainWindow(QWidget *parent=0);
  virtual ~qAmoraDigitalRockAppMainWindow();

public slots:
  void on_HelpAboutAmoraDigitalRockAppAction_triggered();

protected:
  qAmoraDigitalRockAppMainWindow(qAmoraDigitalRockAppMainWindowPrivate* pimpl, QWidget* parent);

private:
  Q_DECLARE_PRIVATE(qAmoraDigitalRockAppMainWindow);
  Q_DISABLE_COPY(qAmoraDigitalRockAppMainWindow);
};

#endif
